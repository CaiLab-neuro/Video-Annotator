<div align="center">

<h2>Annotator Tool of Behavioral Video Data</h2>

Iba Baig (Northeastern University), Mingbo Cai, Ph.D. (University of Miami).

</div>

# Overview

### Abstract

How do children and parents switch attention and choose actions during their dynamic interactions? To study this requires naturalist video recording of their activity. Current research in developmental psychology relies heavily on frame-to-frame manual coding of video recordings, which is time-intensive and subjective. We investigate whether multimodal AI can automatically classify behavioral states during such interaction given language prompts by building a video analysis pipeline employing Tarsier, a large video-language model developed by ByteDance. Our pipeline extracts sequential overlapping brief (1-3 second) video clips from 20-minute videos of child-parent playing activity in the lab, and submits them to the video-language model with standardized prompts for classifying child’s attention direction, hand actions, toy manipulation styles, and parental gesturing behaviors. This process generates a time series of these behavioral features reflecting the moment-to-moment changes of the action and attention of children and parents. Preliminary analysis of the pipeline’s output in comparison to manual annotation confirms its validity. Further, by allowing the model to analyze an entire 20-minute video, we find that the model can reveal quantifiable visual cues such as child’s pose, child’s longest toy play time, distraction, and parental gestures with a single prompt and can also infer the child’s preference among the toys played during the recording session. This approach demonstrates that AI can be utilized for objective measures of social engagement previously inefficient through manual observation and speeds up the analysis of large-scale video datasets of children’s behavior. The methodology has implications for early developmental screening, personalized intervention design, and advancing our understanding of how visual interaction dynamics influence social development. Future work will integrate audio analysis and eye-tracking data.

### Simple Model Structure
Tarsier takes a simple sturcture that use a MLP projection layer to connect visual encoder (CLIP ViT) and text decoder (LLM). Frames are encoded independently and concatenated to input into LLM.

# Usage
This section provides guidance on how to run, evaluate and deploy this model.

## Setup
Following all are running under the environment of python 3.9. First, clone this git repo, then continue to the following instructions to add the Tarsier model.

### Model Prepare
Download the model checkpoints from Hugging Face: [Tarsier2-Recap-7b](https://huggingface.co/omni-research/Tarsier2-Recap-7b).
Complete instructions to integrate Tarsier2 model in folder annotations_module: [Tarsier2](https://github.com/bytedance/tarsier).
