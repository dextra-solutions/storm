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