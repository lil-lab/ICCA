# %%
import pandas as pd
import glob, os, io, random, json, time, argparse, copy, base64, pickle
from datetime import datetime
from PIL import Image
from utils import *
from tqdm import tqdm
from sklearn.metrics import classification_report
from MLLMs import *


def eval_loop(
    trial_entries,
    context_imgs,
    iters,
    random_seed,
    spkr_exp_args,
    lsnr_exp_args,
    spkr_model,
    lsnr_model,
    img_mask,
    img_mask_url=None,
    img_mask_base64=None,
    sleep_time=0,
):
    random.seed(random_seed)
    random_seeds = random.sample(range(0, 1000), iters)
    trials_Records = trial_entries[:iters]

    # prepare the instruction that appears in the beginning of the interaction
    spkr_intro = (
        spkr_model.get_spkr_intro(context_imgs)
        if spkr_exp_args.model_type != "Human"
        else []
    )
    lsnr_intro = (
        lsnr_model.get_lsnr_intro() if lsnr_exp_args.model_type != "oracle" else []
    )

    spkr_context_imgs = context_imgs
    if lsnr_exp_args.img_mask:
        lsnr_context_imgs = copy.deepcopy(context_imgs)
        lsnr_context_imgs = masking_images(
            lsnr_context_imgs, img_mask, img_mask_url, img_mask_base64
        )
    else:
        lsnr_context_imgs = context_imgs.copy()
        random.shuffle(
            lsnr_context_imgs
        )  # speaker and listener see the images in potentially different orders

    for t, R_t in enumerate(trials_Records):
        if t != 0:
            time.sleep(sleep_time)
        tgt_fn = R_t["targetImg"]
        human_msg = R_t["msg"]

        if spkr_exp_args.model_type == "Human":
            gen_msg = human_msg
            spkr_trial_prompt = []
            spkr_prompt = []
        else:
            (
                spkr_prompt,
                spkr_trial_prompt,
                spkr_tgt_img,
                tgt_label_for_spkr,
                spkr_trial_imgs,
            ) = spkr_model.get_spkr_prompt(
                spkr_intro,
                t,
                spkr_context_imgs,
                tgt_fn,
                spkr_exp_args,
                records=trials_Records,
            )

            R_t["spkr_trial_fns"] = [img["filename"] for img in spkr_trial_imgs]

            ##############
            # query the speaker
            if spkr_exp_args.model_type == "llava":
                gen_msg = spkr_model.query(spkr_prompt, spkr_trial_imgs).strip()
            else:
                gen_msg = spkr_model.query(spkr_prompt).strip()

            spkr_trial_prompt = spkr_model.update_with_spkr_pred(
                spkr_trial_prompt, gen_msg
            )

            R_t["spkr_msg"] = gen_msg
            R_t["tgt_label_for_spkr"] = tgt_label_for_spkr

        #############
        # listener prompt prep
        if lsnr_exp_args.model_type == "oracle":
            pred_fn, lsnr_trial_prompt, R_t["tgt_label_for_lsnr"], R_t["lsnr_pred"] = (
                spkr_tgt_img["filename"],
                [],
                tgt_label_for_spkr,
                tgt_label_for_spkr,
            )
            lsnr_prompt = []
            # R_t['lsnr_pred']=lsnr_pred
        else:
            omit_img = True if (lsnr_exp_args.img_once and t > 0) else False

            if t in lsnr_exp_args.misleading_trials:
                misleading = True
                omit_img = False  # override the omit_img
            else:
                misleading = False

            (
                lsnr_prompt,
                lsnr_trial_prompt,
                lsnr_tgt_img,
                tgt_label_for_lsnr,
                lsnr_trial_imgs,
                lsnr_trial_imgs_lsnr_view,
            ) = lsnr_model.get_lsnr_prompt(
                lsnr_intro,
                t,
                lsnr_context_imgs,
                tgt_fn,
                msg=gen_msg,
                records=trials_Records,
                random_seed=random_seeds[t],
                no_history=lsnr_exp_args.no_history,
                do_shuffle=lsnr_exp_args.do_shuffle,
                omit_img=omit_img,
                misleading=misleading,
                has_intro=lsnr_exp_args.has_intro,
            )
            # lsnr_trial_imgs_lsnr_view is different from lsnr_trial_imgs for experiments that have misleading manipulations

            R_t["lsnr_trial_fns"] = [img["filename"] for img in lsnr_trial_imgs]

            ##############
            # query the listener
            if lsnr_exp_args.model_type == "llava":
                lsnr_pred = lsnr_model.query(
                    lsnr_prompt, lsnr_trial_imgs_lsnr_view
                ).lower()
            else:
                lsnr_pred = lsnr_model.query(lsnr_prompt).upper()

            lsnr_pred = lsnr_pred.strip()
            lsnr_trial_prompt = lsnr_model.update_with_lsnr_pred(
                lsnr_trial_prompt, lsnr_pred
            )
            R_t["tgt_label_for_lsnr"] = tgt_label_for_lsnr
            R_t["lsnr_pred"] = lsnr_pred

            try:
                pred_fn = lsnr_trial_imgs[
                    lsnr_model.model_args.label_space.index(lsnr_pred)
                ]["filename"]
            except ValueError:
                pred_fn = "invalid"
            lsnr_feedback = lsnr_model.get_lsnr_feedback(
                lsnr_pred, lsnr_tgt_img, lsnr_trial_imgs, gen_msg
            )
            lsnr_trial_prompt.append(lsnr_feedback)

        # get speaker feedback
        if spkr_exp_args.model_type != "Human":
            spkr_feedback = spkr_model.get_spkr_feedback(
                pred_fn, spkr_tgt_img, spkr_trial_imgs
            )

            spkr_trial_prompt.append(spkr_feedback)

        things_to_print = []

        if spkr_exp_args.model_type != "Human":
            things_to_print.extend(
                [
                    {"Gen_msg": gen_msg},
                    {"Human_msg": human_msg},
                    {"Tgt_fn": spkr_tgt_img["filename"]},
                ]
            )
        else:
            things_to_print.extend([{"Human_msg": human_msg}, {"Tgt_fn": tgt_fn}])

        if lsnr_exp_args.model_type != "oracle":
            things_to_print.extend(
                [
                    {"Pred_fn": pred_fn},
                    {"Pred_label": lsnr_pred},
                    {"Tgt_label": tgt_label_for_lsnr},
                ]
            )

        print(" | ".join([f"{k}: {v}" for d in things_to_print for k, v in d.items()]))

        R_t["spkr_trial_record"] = spkr_trial_prompt
        R_t["lsnr_trial_record"] = lsnr_trial_prompt

    return trials_Records, spkr_prompt, lsnr_prompt


def run_test(
    context_records_fps,
    num_of_trials,
    random_seed,
    save_suffix,
    dtime,
    spkr_exp_args,
    lsnr_exp_args,
    spkr_model,
    lsnr_model,
    img_mask,
    img_mask_url=None,
    img_mask_base64=None,
    sleep_time=0,
    img_hosting_site=None,
    exp_name=None,
):
    random.seed(random_seed)
    seeds = random.sample(range(0, 1000), len(context_records_fps))

    for i, context_record_fp in tqdm(enumerate(context_records_fps)):
        datafile_name = os.path.basename(context_record_fp)

        dirname = os.path.dirname(context_record_fp)
        dirname_save = os.path.join("outputs_icca", dirname)

        print(f"Working on context {datafile_name}")
        trials_this_context = pd.read_csv(context_record_fp, index_col=0)

        img_fps = glob.glob(f"{dirname}/COCO*.jpg")
        if len(img_fps) == 0:
            img_fps = glob.glob(f"{dirname}/*.png")
        context_imgs = []

        for fp in img_fps:
            img = Image.open(fp)
            with open(fp, "rb") as image_file:
                binary_data = image_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode("utf-8")

            fn = os.path.basename(fp)

            if (
                spkr_exp_args.model_type != "Human"
                and spkr_model.model_args.img_mode == "URL"
            ) or (
                lsnr_exp_args.model_type != "oracle"
                and lsnr_model.model_args.img_mode == "URL"
            ):
                url = f"{img_hosting_site}/{fn}"
            else:
                url = None

            entry = {
                "filename": fn,
                "PIL": img,
                "URL": url,
                "base64_string": base64_string,
            }
            context_imgs.append(entry)

        trials_this_context = trials_this_context.to_json(orient="records")
        trials_this_context = json.loads(trials_this_context)

        random.seed(i)
        random.shuffle(context_imgs)

        records, spkr_hist, lsnr_hist = eval_loop(
            trials_this_context,
            context_imgs,
            iters=num_of_trials,
            random_seed=seeds[i],
            spkr_exp_args=spkr_exp_args,
            lsnr_exp_args=lsnr_exp_args,
            spkr_model=spkr_model,
            lsnr_model=lsnr_model,
            img_mask=img_mask,
            img_mask_url=img_mask_url,
            img_mask_base64=img_mask_base64,
            sleep_time=sleep_time,
        )

        records_df = pd.DataFrame(records)
        print(
            classification_report(
                records_df["tgt_label_for_lsnr"],
                records_df["lsnr_pred"],
                zero_division=0,
            )
        )
        report_to_save = classification_report(
            records_df["tgt_label_for_lsnr"],
            records_df["lsnr_pred"],
            zero_division=0,
            output_dict=True,
        )
        full_transcripts = {"spkr_hist": spkr_hist, "lsnr_hist": lsnr_hist}

        spkr_model_args = (
            spkr_model.model_args.__dict__ if spkr_model is not None else None
        )
        lsnr_model_args = (
            lsnr_model.model_args.__dict__ if lsnr_model is not None else None
        )

        report_to_save.update(
            {
                "spkr_model_arg": spkr_model_args,
                "lsnr_model_args": lsnr_model_args,
                "spkr_exp_args": spkr_exp_args.__dict__,
                "lsnr_exp_args": lsnr_exp_args.__dict__,
            }
        )

        os.makedirs(dirname_save, exist_ok=True)
        records_df.to_csv(f"{dirname_save}/records_{dtime}_{save_suffix}_output.csv")
        with open(
            f"{dirname_save}/records_{dtime}_{save_suffix}_report.json", "w"
        ) as f:
            json.dump(report_to_save, f, indent=4)

        with open(
            f"{dirname_save}/records_{dtime}_{save_suffix}_full_transcripts.pickle",
            "wb",
        ) as f:
            pickle.dump(full_transcripts, f)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spkr_model_type", type=str, default="Claude"
    )  # IDEFICS, llava, GPT, Gemini, Claude, Human
    parser.add_argument(
        "--lsnr_model_type", type=str, default="GPT"
    )  # IDEFICS, llava, GPT, Gemini, Claude, oracle
    parser.add_argument(
        "--spkr_intro_version", type=str, default="simple"
    )  # any key in the intro_texts_spkr.json file
    parser.add_argument("--lsnr_intro_version", type=str, default="standard")
    parser.add_argument(
        "--sleep_time", type=int, default=0
    )  # sleep between trials for APIs that have a low rate limit
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="ICCA_data")
    parser.add_argument(
        "--spkr_img_mode", type=str, default="base64_string"
    )  # IDEFICS, llava, and Gemini use PIL as the image format; GPT uses URL; Claude uses base64_string. We will support more input formats for the models soon.
    parser.add_argument("--lsnr_img_mode", type=str, default="URL")
    parser.add_argument("--spkr_API_key", type=str, default="")
    parser.add_argument("--lsnr_API_key", type=str, default="")
    parser.add_argument("--organization_ID", type=str, default="")  # for GPT only
    parser.add_argument(
        "--spkr_model_ckpt", type=str, default="claude-3-opus-20240229"
    )  # HuggingFaceM4/idefics-80b-instruct, liuhaotian/llava-v1.5-13b, gpt-4-1106-vision-preview, gemini-pro-vision
    parser.add_argument(
        "--lsnr_model_ckpt", type=str, default="gpt-4-1106-vision-preview"
    )
    parser.add_argument(
        "--spkr_intro_texts", type=str, default="args/intro_texts_spkr.json"
    )
    parser.add_argument(
        "--lsnr_intro_texts", type=str, default="args/intro_texts_lsnr.json"
    )
    parser.add_argument(
        "--lsnr_exp_args_fp", type=str, default="args/interaction_args_lsnr.json"
    )
    parser.add_argument(
        "--img_hosting_site", type=str, default=""
    )  # to send images to the models as URLs
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument(
        "--num_of_trials", type=int, default=24
    )  # can use smaller numbers for debugging

    args = parser.parse_args()

    with open(args.spkr_intro_texts, "r") as f:
        spkr_intro_texts = json.load(f)

    with open(args.lsnr_intro_texts, "r") as f:
        lsnr_intro_texts = json.load(f)

    spkr_intro_text = (
        None
        if args.spkr_model_type == "Human"
        else spkr_intro_texts[args.spkr_model_type][args.spkr_intro_version]
    )
    lsnr_intro_text = (
        None
        if args.lsnr_model_type == "oracle"
        else lsnr_intro_texts[args.lsnr_model_type][args.lsnr_intro_version]
    )

    with open(args.lsnr_exp_args_fp, "r") as f:
        lsnr_exp_args = json.load(f)

    spkr_exp_args = InteractionArgs(model_type=args.spkr_model_type)
    lsnr_exp_args = InteractionArgs(model_type=args.lsnr_model_type, **lsnr_exp_args)

    print("initializing spkr model..")
    match spkr_exp_args.model_type:
        case "llava":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                img_mode=args.spkr_img_mode,
                max_output_tokens=30,
                label_space=["top left", "top right", "bottom left", "bottom right"],
                intro_text=spkr_intro_text,
            )
            spkr_model = LlavaModel(spkr_model_args)

        case "IDEFICS":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                img_mode=args.spkr_img_mode,
                max_output_tokens=30,
                label_space=["A", "B", "C", "D"],
                intro_text=spkr_intro_text,
            )
            spkr_model = IDEFICSModel(spkr_model_args)

        case "Gemini":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                img_mode=args.spkr_img_mode,
                max_output_tokens=30,
                label_space=["A", "B", "C", "D"],
                intro_text=spkr_intro_text,
            )
            spkr_model = GeminiModel(spkr_model_args, API_key=args.spkr_API_key)
        case "GPT":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                img_mode=args.spkr_img_mode,
                max_output_tokens=30,
                label_space=["A", "B", "C", "D"],
                intro_text=spkr_intro_text,
            )
            spkr_model = GPTModel(
                spkr_model_args,
                organization_ID=args.organization_ID,
                API_key=args.spkr_API_key,
            )

        case "Claude":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                img_mode=args.spkr_img_mode,
                max_output_tokens=30,
                label_space=["A", "B", "C", "D"],
                intro_text=spkr_intro_text,
            )
            spkr_model = ClaudeModel(spkr_model_args, API_key=args.spkr_API_key)

        case "Human":
            spkr_model = None

    print("initializing lsnr model..")
    match lsnr_exp_args.model_type:
        case "IDEFICS":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=1,
                img_mode=args.lsnr_img_mode,
                label_space=["A", "B", "C", "D"],
                intro_text=lsnr_intro_text,
            )
            lsnr_model = IDEFICSModel(lsnr_model_args)

        case "llava":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=5,
                img_mode=args.lsnr_img_mode,
                label_space=["top left", "top right", "bottom left", "bottom right"],
                intro_text=lsnr_intro_text,
            )
            lsnr_model = LlavaModel(lsnr_model_args)

        case "GPT":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=1,
                img_mode=args.lsnr_img_mode,
                label_space=["A", "B", "C", "D"],
                intro_text=lsnr_intro_text,
            )
            lsnr_model = GPTModel(
                lsnr_model_args,
                organization_ID=args.organization_ID,
                API_key=args.lsnr_API_key,
            )
        case "Gemini":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=5,
                img_mode=args.lsnr_img_mode,
                label_space=["A", "B", "C", "D"],
                intro_text=lsnr_intro_text,
            )
            lsnr_model = GeminiModel(lsnr_model_args, API_key=args.lsnr_API_key)

        case "Claude":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=1,
                img_mode=args.lsnr_img_mode,
                label_space=["A", "B", "C", "D"],
                intro_text=lsnr_intro_text,
            )
            lsnr_model = ClaudeModel(lsnr_model_args, API_key=args.lsnr_API_key)

        case "oracle":
            lsnr_model = None

    if lsnr_exp_args.img_mask:
        img_mask = Image.new("RGB", (256, 256), color=(0, 0, 0))
        buffer = io.BytesIO()
        img_mask.save(buffer, format="JPEG")
        buffer.seek(0)
        img_mask = Image.open(buffer)
        img_mask_url = f"{args.img_hosting_site}/black_mask.jpg"
        img_mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    else:
        img_mask = None
        img_mask_url = None
        img_mask_base64 = None

    print(
        f"spkr: {spkr_exp_args.model_type} | lsnr: {lsnr_exp_args.model_type} | spkr_intro: {args.spkr_intro_version} | lsnr_intro: {args.lsnr_intro_version}"
    )
    save_suffix = f"{spkr_exp_args.model_type}_{args.spkr_intro_version}_{lsnr_exp_args.model_type}_{args.lsnr_intro_version}"

    test_dir = args.data_dir
    context_records_fps = glob.glob(f"{test_dir}/*/trials_for*.csv")
    context_records_fps.sort()

    dtime = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_test(
        context_records_fps,
        num_of_trials=args.num_of_trials,
        random_seed=args.seed,
        save_suffix=save_suffix,
        spkr_exp_args=spkr_exp_args,
        dtime=dtime,
        lsnr_exp_args=lsnr_exp_args,
        spkr_model=spkr_model,
        lsnr_model=lsnr_model,
        img_mask=img_mask,
        img_mask_url=img_mask_url,
        img_mask_base64=img_mask_base64,
        sleep_time=args.sleep_time,
        img_hosting_site=args.img_hosting_site,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
