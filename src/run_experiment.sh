python run_icca.py \
    --spkr_model_type "Claude" \
    --lsnr_model_type "GPT" \
    --spkr_intro_version "simple" \
    --lsnr_intro_version "standard" \
    --data_dir "icca_data" \
    --spkr_img_mode "base64_string" \
    --lsnr_img_mode "URL" \
    --spkr_API_key "" \
    --lsnr_API_key "" \
    --organization_ID "" \
    --spkr_model_ckpt "claude-3-opus-20240229" \
    --lsnr_model_ckpt "gpt-4-1106-vision-preview" \
    --spkr_intro_texts "args/intro_texts_spkr.json" \
    --lsnr_intro_texts "args/intro_texts_lsnr.json" \
    --lsnr_exp_args_fp "args/interaction_args_lsnr.json" \
    



#spkr_intro_version and lsnr_intro_version specify the exact prompts to be used for the models. See intro_texts_spkr.json and intro_texts_lsnr.json for examples.

#for listener experiments, we keep the listener instruction mostly consistent and vary how the conversation history is presented. The only listener variant for which we modified the instruction slightly is L5 Images masked. Models may refuse to output a valid image label under the standard listener prompt because of image masking. The exception is Gemini, which always consistently output a valid image label, so we still used the standard listener prompt for it. 