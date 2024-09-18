from itertools import chain
from PIL import Image
import time, copy, random
import openai
from openai import OpenAI
from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
    APITimeoutError,
)
import httpx
from abc import ABC, abstractmethod
from anthropic import Anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class ModelWrapper(ABC):
    def __init__(self, model_args):
        self.model_args = model_args
        print("model_ckpt:", model_args.model_ckpt)

    @abstractmethod
    def get_spkr_intro(self, context_imgs):
        pass

    @abstractmethod
    def get_lsnr_intro(self):
        pass

    def _model_specific_prompt_postprocessing(self, prompt):
        return prompt  # can override this as needed

    def get_spkr_prompt(
        self, intro, t, context_imgs, target_fn, interaction_args, records=[]
    ):
        label_space = self.model_args.label_space
        history = [entry["spkr_trial_record"] for entry in records[:t]] if t > 0 else []
        for i in range(4):
            if context_imgs[i]["filename"] == target_fn:
                target_label = label_space[i]
                target_img = context_imgs[i]
                break

        if interaction_args.no_history:
            round_name = "Current Round"
        else:
            round_name = f"Round {t+1}"

        trial_prompt = self._get_spkr_prompt(round_name, target_label)

        history = list(chain.from_iterable(history))

        if intro not in history and interaction_args.has_intro:
            history = intro + history

        prompt = history + trial_prompt

        prompt = self._model_specific_prompt_postprocessing(prompt)
        return prompt, trial_prompt, target_img, target_label, context_imgs

    def get_lsnr_prompt(
        self,
        intro,
        t,
        context_imgs,
        target_fn,
        msg,
        records=[],
        random_seed=None,
        no_history=False,
        do_shuffle=False,
        omit_img=False,
        misleading=False,
        has_intro=True,
    ):
        if no_history:
            history = []

        else:
            history = (
                [entry["lsnr_trial_record"] for entry in records[:t]] if t > 0 else []
            )

        trial_imgs = context_imgs.copy()
        if do_shuffle:
            random.seed(random_seed)
            random.shuffle(trial_imgs)

        label_space = self.model_args.label_space
        for i in range(4):
            if trial_imgs[i]["filename"] == target_fn:
                target_img = trial_imgs[i]
                target_label = label_space[i]
                break

        trial_imgs_after_1st_shuffle = trial_imgs.copy()  # in the misleading manipulation, the gold label will be based on this order but the images are shuffled again

        if misleading:
            random.seed(t + 1)
            random.shuffle(trial_imgs)

        if no_history:
            round_name = "Current Round"
        else:
            round_name = f"Round {t+1}"

        trial_prompt = self._get_lsnr_prompt(round_name, trial_imgs, msg, omit_img)

        history = list(chain.from_iterable(history))

        if intro not in history and has_intro:
            history = intro + history

        prompt = history + trial_prompt

        prompt = self._model_specific_prompt_postprocessing(prompt)
        return (
            prompt,
            trial_prompt,
            target_img,
            target_label,
            trial_imgs_after_1st_shuffle,
            trial_imgs,
        )

    @abstractmethod
    def _get_spkr_prompt(self, round_name, target_label):
        pass

    @abstractmethod
    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        pass

    @abstractmethod
    def query(self, query):
        pass

    @abstractmethod
    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        pass

    @abstractmethod
    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        pass

    # how the feedback is presented is model-dependent and can be thought of as a hyperparameter. can try different phrasing/formats.
    def get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        return self._get_spkr_feedback(pred_fn, spkr_tgt_img, spkr_trial_imgs)

    def get_lsnr_feedback(self, pred, target_img, context_imgs, spkr_msg):
        target_label = self.model_args.label_space[
            [img["filename"] for img in context_imgs].index(target_img["filename"])
        ]
        return self._get_lsnr_feedback(pred, target_label, spkr_msg)

    @abstractmethod
    def _get_lsnr_feedback(self, pred, target_label, spkr_msg):
        pass


class IDEFICSModel(ModelWrapper):
    def __init__(self, model_args, loaded_model=None):
        super().__init__(model_args)
        import torch
        from transformers import (
            IdeficsForVisionText2Text,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = model_args.model_ckpt
        if checkpoint == "HuggingFaceM4/idefics-9b-instruct":
            self.model = IdeficsForVisionText2Text.from_pretrained(
                checkpoint, torch_dtype=torch.bfloat16
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            self.exit_condition = self.processor.tokenizer(
                "<end_of_utterance>", add_special_tokens=False
            ).input_ids
            self.bad_words_ids = self.processor.tokenizer(
                ["<image>", "<fake_token_around_image>"], add_special_tokens=False
            ).input_ids
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_8bit_compute_dtype="float16"
            )
            self.model = IdeficsForVisionText2Text.from_pretrained(
                checkpoint, quantization_config=quantization_config, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            self.exit_condition = self.processor.tokenizer(
                "<end_of_utterance>", add_special_tokens=False
            ).input_ids
            self.bad_words_ids = self.processor.tokenizer(
                ["<image>", "<fake_token_around_image>"], add_special_tokens=False
            ).input_ids

    def query(self, query):
        split_pattern = copy.copy(query[-1])
        inputs = self.processor(query, return_tensors="pt").to(self.device)
        generation_output = self.model.generate(
            **inputs,
            eos_token_id=self.exit_condition,
            bad_words_ids=self.bad_words_ids,
            max_new_tokens=self.model_args.max_output_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(
            generation_output.sequences, skip_special_tokens=True
        )
        gen_msg = generated_text[-1].split(split_pattern)[-1]
        return gen_msg

    def get_spkr_intro(self, context_imgs):
        intro_text = self.model_args.intro_text
        targetnames = self.model_args.label_space
        mode = self.model_args.img_mode
        intro = [
            intro_text,
            f"\nHere are the images:",
            f"\nImage {targetnames[0]}: ",
            context_imgs[0][mode],
            f"\nImage {targetnames[1]}: ",
            context_imgs[1][mode],
            f"\nImage {targetnames[2]}: ",
            context_imgs[2][mode],
            f"\nImage {targetnames[3]}: ",
            context_imgs[3][mode],
            "<end_of_utterance>",
        ]

        return intro

    def _get_spkr_prompt(self, round_name, target_label):
        prompt = [
            f"\nUser: {round_name}, target is Image {target_label}.<end_of_utterance>",
            "\nAssistant: message: ",
        ]
        return prompt

    def get_lsnr_intro(self):
        return [self.model_args.intro_text]

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        label_space = self.model_args.label_space
        mode = self.model_args.img_mode
        if omit_img:
            prompt = [
                f"\nUser: {round_name}",
                f"\nWhich image is this message referring to: {msg}?<end_of_utterance>",
                "\nAssistant: Image ",
            ]
        else:
            prompt = [
                f"\nUser: {round_name}",
                f"User: For each image, remember its label and answer the question below with an image label.",
                f"\nImage {label_space[0]}:",
                trial_imgs[0][mode],
                f"\nImage {label_space[1]}:",
                trial_imgs[1][mode],
                f"\nImage {label_space[2]}:",
                trial_imgs[2][mode],
                f"\nImage {label_space[3]}:",
                trial_imgs[3][mode],
                f"\nWhich image is this message referring to: {msg}?<end_of_utterance>",
                "\nAssistant: Image ",
            ]

        return prompt

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_trial_prompt[-1] = spkr_trial_prompt[-1] + spkr_pred + "<end_of_utterance>"
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        lsnr_trial_prompt[-1] = lsnr_trial_prompt[-1] + lsnr_pred + "<end_of_utterance>"
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        if pred_fn == "invalid":
            feedback = (
                "\nUser: the listener didn't give a valid answer.<end_of_utterance>"
            )
        else:
            for i in range(4):
                if spkr_trial_imgs[i]["filename"] == pred_fn:
                    pred_label = self.model_args.label_space[i]
                    break

            if pred_fn == spkr_tgt_img["filename"]:
                feedback = f"\nUser: the listener correctly answered Image {pred_label}.<end_of_utterance>"
            else:
                feedback = f"\nUser: the listener mistakenly answered Image {pred_label}.<end_of_utterance>"

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg=None):
        if pred == target_label:
            feedback = f"\nUser: Correct.<end_of_utterance>"

        elif pred not in self.model_args.label_space:
            feedback = f"\nUser: Invalid answer. Answer must be one of {self.model_args.label_space}.<end_of_utterance>"
        else:
            feedback = (
                f"\nUser: Wrong, the answer is Image {target_label}.<end_of_utterance>"
            )

        return feedback


class LlavaModel(ModelWrapper):
    def __init__(self, model_args, loaded_model=None):
        self.model_args = model_args
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.eval.run_llava import eval_model

        self.eval_model = eval_model

        model_path = model_args.model_ckpt

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path, model_base=None, model_name=self.model_name
            )
        )

    def query(self, query, trial_imgs):
        query = "".join(query)

        merged_context = merge_images(
            [img[self.model_args.img_mode] for img in trial_imgs]
        )
        args = type(
            "Args",
            (),
            {
                "model_name": self.model_name,
                "model": self.model,
                "context_len": self.context_len,
                "tokenizer": self.tokenizer,
                "image_processor": self.image_processor,
                "query": query,
                "conv_mode": None,
                "sep": ",",
                "images": [merged_context],
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.model_args.max_output_tokens,
            },
        )()

        gen_msg, _ = self.eval_model(args)

        return gen_msg

    def get_spkr_intro(self, context_imgs):
        return [self.model_args.intro_text]

    def _get_spkr_prompt(self, round_name, target_label):
        prompt = [f"\n{round_name}", f"\nTarget: {target_label}. Message:"]
        return prompt

    def get_lsnr_intro(self):
        return [self.model_args.intro_text]

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        prompt = [
            f"\n{round_name}",
            f"\nWhich image is this message referring to: {msg}?",
        ]

        return prompt

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_trial_prompt[-1] = (
            spkr_trial_prompt[-1] + " ASSISTANT: " + spkr_pred + "</s></s>"
        )
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        lsnr_trial_prompt[-1] = (
            lsnr_trial_prompt[-1] + " ASSISTANT: " + lsnr_pred + "</s></s>"
        )
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        if pred_fn == "invalid":
            feedback = "USER: the listener didn't give a valid answer."
        else:
            for i in range(4):
                if spkr_trial_imgs[i]["filename"] == pred_fn:
                    pred_label = self.model_args.label_space[i]
                    break

            if pred_fn == spkr_tgt_img["filename"]:
                feedback = f"USER: the listener correctly answered {pred_label}."
            else:
                feedback = f"USER: the listener mistakenly answered {pred_label}."

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg):
        if pred == target_label:
            feedback = "USER: correct."
        elif pred not in self.model_args.label_space:
            feedback = "USER: invalid answer. Answer must be one of top left, top right, bottom left, bottom right."
        else:
            feedback = f"USER: wrong, {spkr_msg} is referring to {target_label}."

        return feedback


class GPTModel(ModelWrapper):
    def __init__(self, model_args, organization_ID, API_key, client=None):
        super().__init__(model_args)
        openai.organization = organization_ID
        openai.api_key = API_key
        self.client = OpenAI(
            api_key=openai.api_key,
            organization=openai.organization,
            timeout=httpx.Timeout(15.0, read=5.0, write=10.0, connect=3.0),
        )
        self.spkr_system_msg = {
            "role": "system",
            "content": "you are a smart assistant that will follow the user's instructions",
        }
        self.lsnr_system_msg = {
            "role": "system",
            "content": "You are an assistant who will play a series of reference games with the user. You will pay close attention to the conversation history as more rounds are played.",
        }
        self.enforce_alternating_roles = False
        self.retry_after_seconds = 15
        self.retry_limit = 10
        assert (
            model_args.img_mode == "URL"
        ), "change the interaction template to support other input image types"

    def get_spkr_intro(self, context_imgs):
        intro_text = self.model_args.intro_text
        targetnames = self.model_args.label_space
        intro = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": intro_text},
                    {"type": "text", "text": f"\nImage {targetnames[0]}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": context_imgs[0]["URL"], "detail": "low"},
                    },
                    {"type": "text", "text": f"\nImage {targetnames[1]}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": context_imgs[1]["URL"], "detail": "low"},
                    },
                    {"type": "text", "text": f"\nImage {targetnames[2]}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": context_imgs[2]["URL"], "detail": "low"},
                    },
                    {"type": "text", "text": f"\nImage {targetnames[3]}: "},
                    {
                        "type": "image_url",
                        "image_url": {"url": context_imgs[3]["URL"], "detail": "low"},
                    },
                ],
            }
        ]

        return intro

    def _get_spkr_prompt(self, round_name, target_label):
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{round_name}, "},
                    {"type": "text", "text": f"\nThe target is Image {target_label}."},
                ],
            }
        ]
        return prompt

    def get_lsnr_intro(self):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.model_args.intro_text}],
            }
        ]

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        label_space = self.model_args.label_space
        if omit_img:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{round_name}, "},
                        {
                            "type": "text",
                            "text": f"\nWhich image is this message referring to: {msg}? Output the image label only (a single letter).",
                        },
                    ],
                }
            ]

        else:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{round_name}, "},
                        {"type": "text", "text": f"\nImage {label_space[0]}: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": trial_imgs[0]["URL"], "detail": "low"},
                        },
                        {"type": "text", "text": f"\nImage {label_space[1]}: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": trial_imgs[1]["URL"], "detail": "low"},
                        },
                        {"type": "text", "text": f"\nImage {label_space[2]}: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": trial_imgs[2]["URL"], "detail": "low"},
                        },
                        {"type": "text", "text": f"\nImage {label_space[3]}: "},
                        {
                            "type": "image_url",
                            "image_url": {"url": trial_imgs[3]["URL"], "detail": "low"},
                        },
                        {
                            "type": "text",
                            "text": f"\nWhich image is this message referring to: {msg}? Output the image label only (a single letter).",
                        },
                    ],
                }
            ]

        return prompt

    def query(self, query):
        response = self._gpt_query(query)
        gen_msg = response.choices[0].message.content
        return gen_msg

    def _gpt_query(self, query, times_retried=0):
        if times_retried > self.retry_limit:
            raise Exception("retry failed")

        system_msg = (
            self.spkr_system_msg
            if self.model_args.role == "spkr"
            else self.lsnr_system_msg
        )
        messages = [system_msg] + query

        try:
            response = self.client.chat.completions.create(
                model=self.model_args.model_ckpt,
                messages=messages,
                seed=42,
                max_tokens=self.model_args.max_output_tokens,
                temperature=0,
                timeout=60,
            )

        except APIError as e:
            print(e)
            print(f"retrying in {self.retry_after_seconds} seconds")
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)

        except RateLimitError as e:
            print(
                f"Rate limit exceeded. Waiting and retrying in {self.retry_after_seconds} seconds..."
            )
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)
        except APITimeoutError as e:
            print(e)
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)

        except Exception as e:
            print(e)
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)

        return response

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_pred_formatted = {"role": "assistant", "content": spkr_pred}
        spkr_trial_prompt.append(spkr_pred_formatted)
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        assistant_pred = {"role": "assistant", "content": lsnr_pred}
        lsnr_trial_prompt.append(assistant_pred)
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        if pred_fn == "invalid":
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The listener didn't give a valid answer.",
                    }
                ],
            }
        else:
            for i in range(4):
                if spkr_trial_imgs[i]["filename"] == pred_fn:
                    pred_label = self.model_args.label_space[i]
                    break

            if pred_fn == spkr_tgt_img["filename"]:
                feedback = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The listener correctly answered Image {pred_label}.",
                        }
                    ],
                }
            else:
                feedback = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The listener mistakenly answered Image {pred_label}.",
                        }
                    ],
                }

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg=None):
        if pred == target_label:
            feedback = {
                "role": "user",
                "content": [{"type": "text", "text": f"Correct."}],
            }
        elif pred not in self.model_args.label_space:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Invalid answer. Answer must be one of {self.model_args.label_space}.",
                    }
                ],
            }
        else:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Wrong, I'm referring to Image {target_label}.",
                    }
                ],
            }

        return feedback

    def _model_specific_prompt_postprocessing(self, prompt):
        if self.enforce_alternating_roles:
            prompt = copy.deepcopy(prompt)
            formatted_prompt = [prompt[0]]
            for i in range(1, len(prompt)):
                if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                    prompt[i]["content"][0]["text"] = (
                        "\n" + prompt[i]["content"][0]["text"]
                    )
                    formatted_prompt[-1]["content"] = (
                        formatted_prompt[-1]["content"] + prompt[i]["content"]
                    )
                else:
                    formatted_prompt.append(prompt[i])

            return formatted_prompt
        return prompt


class GeminiModel(ModelWrapper):
    def __init__(self, model_args, API_key):
        super().__init__(model_args)
        genai.configure(api_key=API_key)
        self.model = genai.GenerativeModel(model_args.model_ckpt)
        self.gemini_config = genai.GenerationConfig(
            candidate_count=1,
            stop_sequences=["\n", "."],
            top_k=1,
            max_output_tokens=model_args.max_output_tokens,
            temperature=0,
        )

        self.retry_after_seconds = 15
        self.retry_limit = 10

    def get_spkr_intro(self, context_imgs):
        intro_text = self.model_args.intro_text
        mode = self.model_args.img_mode
        targetnames = self.model_args.label_space
        intro = [
            intro_text,
            f"\nHere are the images.",
            f"\nImage {targetnames[0]}: ",
            context_imgs[0][mode],
            f"\nImage {targetnames[1]}: ",
            context_imgs[1][mode],
            f"\nImage {targetnames[2]}: ",
            context_imgs[2][mode],
            f"\nImage {targetnames[3]}: ",
            context_imgs[3][mode],
        ]
        return intro

    def _get_spkr_prompt(self, round_name, target_label):
        prompt = [
            f"\nsystem: {round_name}, target is Image {target_label}.",
            "\nspeaker: ",
        ]
        return prompt

    def get_lsnr_intro(self):
        return [self.model_args.intro_text]

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        label_space = self.model_args.label_space
        img_mode = self.model_args.img_mode
        if omit_img:
            prompt = [
                f"\nsystem: {round_name}",
                f"\nWhich image is this message referring to: {msg}?",
                f"\nlistener: ",
            ]
        else:
            prompt = [
                f"\nsystem: {round_name}",
                f"\nImage {label_space[0]}:",
                trial_imgs[0][img_mode],
                f"\nImage {label_space[1]}:",
                trial_imgs[1][img_mode],
                f"\nImage {label_space[2]}:",
                trial_imgs[2][img_mode],
                f"\nImage {label_space[3]}:",
                trial_imgs[3][img_mode],
                f"\nWhich image is this message referring to: {msg}?",
                f"\nlistener: ",
            ]

        return prompt

    def query(self, query):
        gen_msg = self._gemini_query(query)
        return gen_msg

    def _gemini_query(self, query, times_retried=0):
        if times_retried > self.retry_limit:
            raise Exception("retry failed")

        try:
            response = self.model.generate_content(
                query,
                generation_config=self.gemini_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            return response.text
        except Exception as e:
            print(e, flush=True)
            print(f"retrying in {self.retry_after_seconds} seconds...")
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gemini_query(query, times_retried=times_retried)

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_trial_prompt[-1] += spkr_pred
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        lsnr_trial_prompt[-1] += lsnr_pred
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        if pred_fn == "invalid":
            feedback = f"\nsystem: The listener didn't give a valid answer."
        else:
            for i in range(4):
                if spkr_trial_imgs[i]["filename"] == pred_fn:
                    pred_label = self.model_args.label_space[i]
                    break

            if pred_fn == spkr_tgt_img["filename"]:
                feedback = (
                    f"\nsystem: The listener correctly answered Image {pred_label}."
                )
            else:
                feedback = (
                    f"\nsystem: The listener mistakenly answered Image {pred_label}."
                )

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg=None):
        if pred == target_label:
            feedback = f"\nsystem: Correct. I'm referring to Image {target_label}."
        elif pred not in self.model_args.label_space:
            feedback = f"\nsystem: Invalid answer. Answer must be one of {self.model_args.label_space}."
        else:
            feedback = f"\nsystem: Wrong. I'm referring to Image {target_label}."

        return feedback


class ClaudeModel(ModelWrapper):
    def __init__(self, model_args, API_key):
        super().__init__(model_args)
        self.client = Anthropic(api_key=API_key)

        self.lsnr_system_msg = "You are an assistant who will play a series of reference games with the user. You will pay close attention to the conversation history as more rounds are played."

    def get_spkr_intro(self, context_imgs):
        intro_text = self.model_args.intro_text
        label_space = self.model_args.label_space
        mode = self.model_args.img_mode
        intro = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": intro_text},
                    {"type": "text", "text": f"Image {label_space[0]}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": context_imgs[0][mode],
                        },
                    },
                    {"type": "text", "text": f"Image {label_space[1]}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": context_imgs[1][mode],
                        },
                    },
                    {"type": "text", "text": f"Image {label_space[2]}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": context_imgs[2][mode],
                        },
                    },
                    {"type": "text", "text": f"Image {label_space[3]}:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": context_imgs[3][mode],
                        },
                    },
                ],
            }
        ]

        return intro

    def _get_spkr_prompt(self, round_name, target_label):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{round_name}, the target is Image {target_label}.",
                    }
                ],
            }
        ]

        return prompt

    def get_lsnr_intro(self):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.model_args.intro_text}],
            }
        ]

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        label_space = self.model_args.label_space
        mode = self.model_args.img_mode
        if omit_img:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{round_name}, "},
                        {
                            "type": "text",
                            "text": f"which image is this message referring to: {msg}?",
                        },
                    ],
                },
                {"role": "assistant", "content": "Image"},
            ]

        else:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{round_name}, "},
                        {"type": "text", "text": f"Image {label_space[0]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[0][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[1]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[1][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[2]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[2][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[3]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[3][mode],
                            },
                        },
                        {
                            "type": "text",
                            "text": f"which image is this message referring to: {msg}?",
                        },
                    ],
                },
                {"role": "assistant", "content": "Image"},
            ]

        return prompt

    def query(self, query):
        if self.model_args.role == "lsnr":
            response = self.client.messages.create(
                model=self.model_args.model_ckpt,
                max_tokens=self.model_args.max_output_tokens,
                messages=query,
                temperature=0,
                top_k=1,
                system=self.lsnr_system_msg,
            )
        else:
            response = self.client.messages.create(
                model=self.model_args.model_ckpt,
                max_tokens=self.model_args.max_output_tokens,
                messages=query,
                temperature=0,
                top_k=1,
            )

        return response.content[0].text

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_pred_formatted = {
            "role": "assistant",
            "content": [{"type": "text", "text": spkr_pred}],
        }
        spkr_trial_prompt.append(spkr_pred_formatted)
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        lsnr_trial_prompt[-1]["content"] = (
            lsnr_trial_prompt[-1]["content"] + " " + lsnr_pred
        )
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        if pred_fn == "invalid":
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The listener didn't give a valid answer.",
                    }
                ],
            }
        else:
            for i in range(4):
                if spkr_trial_imgs[i]["filename"] == pred_fn:
                    pred_label = self.model_args.label_space[i]
                    break

            if pred_fn == spkr_tgt_img["filename"]:
                feedback = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The listener correctly answered Image {pred_label}.",
                        }
                    ],
                }
            else:
                feedback = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The listener mistakenly answered Image {pred_label}.",
                        }
                    ],
                }

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg=None):
        if pred == target_label:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Correct, I'm referring to Image {target_label}.",
                    }
                ],
            }
        elif pred not in self.model_args.label_space:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Invalid answer. Answer must be one of {self.model_args.label_space}.",
                    }
                ],
            }
        else:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Wrong, I'm referring to Image {target_label}.",
                    }
                ],
            }

        return feedback

    def _model_specific_prompt_postprocessing(self, prompt):
        prompt = copy.deepcopy(prompt)
        formatted_prompt = [prompt[0]]
        for i in range(1, len(prompt)):
            if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                prompt[i]["content"][0]["text"] = "\n" + prompt[i]["content"][0]["text"]
                formatted_prompt[-1]["content"] = (
                    formatted_prompt[-1]["content"] + prompt[i]["content"]
                )
            else:
                formatted_prompt.append(prompt[i])

        return formatted_prompt


def merge_images(Imgs, padding=2, dim=256):
    collage = Image.new("RGB", (dim * 2 + padding, dim * 2 + padding), color=(0, 0, 0))
    collage.paste(Imgs[0], (0, 0))
    collage.paste(Imgs[1], (dim + 2 * padding, 0))
    collage.paste(Imgs[2], (0, dim + 2 * padding))
    collage.paste(Imgs[3], (dim + 2 * padding, dim + 2 * padding))

    return collage
