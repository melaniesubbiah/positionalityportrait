# Positionality Portrait

Code associated with the paper: <link-coming-soon>

This code generates a positionality portrait for an LLM in relation to a dataset. We include the outputs necessary to replicate the results reported in the paper. You can re-generate the paper plots using the notebooks directory and generate_portrait.sh script. 

To run your own portrait, first create summaries for your documents using ≥5 random seeds. We include a sample summarizaiton script in scripts/summarization.py but you will need to load your own documents where indicated in the file. You can modify the code to handle additional demographic groups as well. You can modify the summarization prompts at the top of the file as well.

Summary files with explit demographics in the prompt should be called {model}_with_demo_{seed}.csv with the fields: ["interview_id", "interview", "interviewee_responses", "summary_full", "summary_main", "summary_values", 'gender', 'race']

Baseline summaries should be called {model}_baseline_{seed}.csv with the fields: ["interview_id", "interview", "interviewee_responses", "summary_full", "summary_main", "summary_values", 'gender', 'race']

Once you have all of your summary files in a 'summary' directory, you can generate a portrait for a given model by running generate_portrait.sh (and specify the model you want at the top of the file). You can delete the 'summary_semantics', 'summary_emotions', and 'summary_values' directories we have included unless you are trying to replicate our results.

## Data
The data directory includes resources for the LIWC lexicon, VAD lexicon, and SCM dictionary. Refer the original publications for the associated repositories and citations.

## Citation
<coming-soon>

