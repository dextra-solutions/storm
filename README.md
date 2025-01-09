STORM Wiki pipeline powered by a VLLM server and a search engine.
Required to set up the following environment variables to run this script:
    - `INFERENCE_API_KEY`: API key for the inference model.
    - `SEARCH_API_KEY`: API key for the search engine.

Example usage from a Python script:
```python
from run_storm_wiki_vllm import storm

storm(topic="wireless capable audiophile headsets",
      inference_url="https://api.studio.nebius.ai",
      inference_port=443,
      inference_model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
      output_dir="./results/output",
      max_conv_turn=3,
      max_perspective=3,
      search_top_k=3,
      max_thread_num=3,
      retriever="serper",
      retrieve_top_k=3,
      do_research=True,
      do_generate_outline=True,
      do_generate_article=True,
      do_polish_article=True,
      remove_duplicate=False)
```


Example command line usage:
```bash
python run_storm_wiki_vllm.py --url https://api.studio.nebius.ai --port 443 --model "meta-llama/Meta-Llama-3.1-70B-Instruct-fast" --retriever serper --output-dir ./results/output --do-research --do-generate-outline --do-generate-article --do-polish-article
```

Note: the topic of the article is hardcoded in this example script. To change the topic, modify the `topic` variable in the script.

