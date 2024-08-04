# Evaluating In-context Conversational Adaptation in Multimodal LLMs
This is the official repo for our COLM 2024 paper “Talk Less, Interact Better: Evaluating In-context Conversational Adaptation in Multimodal LLMs”

Yilun Hua and Yoav Artzi

## Prerequisites

Please follow the official guides of the multimodal LLMs (MLLMs) to set up their respective environments. ICCA’s current codebase supports evaluation of IDEFICS, LLaVa, GPT4, Gemini, and Claude. ICCA can also be easily customized to evaluate more MLLMs (see our instructions below).

Additionally, ICCA requires python 3.10, sklearn, and pillow 

## Running ICCA

Edit the arguments in `run_experiment.sh` to specify the speaker and listener models and their respective checkpoint names. For GPT, Gemini, and Claude, also specify your API key and organization key (if applicable).

See `run_icca.py` for the example input values of the arguments.  



## Interaction customization

Section 4, 5 and Appendix D of our paper explain the reference game variants we designed for evaluating MLLMs. These variants can be easily achieved by ICCA by modifying their respective arguments. 

**Speaker Experiments**

You can control what prompt the speaker model receives using the `intro_texts_spkr.json`, which currently shows examples of soliciting efficient communication with instructions of varied levels of explicitness. 



**Listener Experiments** 

You can modify the arguments in `interaction_args_lsnr.json` for different game variants introduced in our paper, including hiding the images after Trial 1, controlling whether to shuffle between trials, when to introduce misleading images, and etc. You can also modify the instructions given to the model in `intro_texts_lsnr.json`.



## Model-specific customization

You can customize the models in `MLLMs.py`. You can change the interaction template, such as how the model receives the image, the phrasing of the system feedback, text generation hyperparameters, and etc. 



## Evaluating new models 

For each new model, you can implement a new wrapper class that inherits from the `ModelWrapper` class in `MLLMs.py`. You can then conduct prompt engineering and model-specific customization following the instructions above. 
