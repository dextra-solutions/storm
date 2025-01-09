"""
STORM Wiki pipeline powered by a VLLM server and a search engine.
You need to set up the following environment variables to run this script:
    - INFERENCE_API_KEY: API key for the inference model.
    - SEARCH_API_KEY: API key for the search engine.

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""
import os
from argparse import ArgumentParser

from dspy import Example

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import VLLMClient
from knowledge_storm.rm import YouRM, BingSearch, BraveRM, SerperRM, DuckDuckGoSearchRM, TavilySearchRM, SearXNG


def storm(topic: str,  ## topic to research
          inference_model: str, inference_url: str, inference_port: int,  ## llm inference related arguments
          output_dir: str,  ## Output directory
          max_conv_turn: int, max_perspective: int, search_top_k: int, max_thread_num: int,  ## hyperparameters for the pre-writing stage
          retriever: str, retrieve_top_k: int,  ##  search related arguments
          do_research: bool, do_generate_outline: bool, do_generate_article: bool, do_polish_article: bool,  ## pipeline stage related arguments
          remove_duplicate: bool  ## polish step extra argument
          ):
    """
    Run the STORM Wiki pipeline using an inference engine hosted by VLLM and a search engine API.
    :param topic: The topic to research.
    :param inference_model: The model to use for inference.
    :param inference_url: The URL of the VLLM server.
    :param inference_port: The port of the VLLM server.
    :param output_dir: The directory to store the outputs.
    :param max_conv_turn: Maximum number of questions in conversational question asking.
    :param max_perspective: Maximum number of perspectives to consider in perspective-guided question asking.
    :param search_top_k: Top k search results to consider for each search query.
    :param max_thread_num: Maximum number of threads to use. The information seeking part and the article generation
                            part can speed up by using multiple threads. Consider reducing it if keep getting "Exceed rate limit"
                            error when calling LM API.
    :param retriever: The search engine API to use for retrieving information.
    :param retrieve_top_k: Top k collected references for each section title.
    :param do_research: If True, simulate conversation to research the topic; otherwise, load the results.
    :param do_generate_outline: If True, generate an outline for the topic; otherwise, load the results.
    :param do_generate_article: If True, generate an article for the topic; otherwise, load the results.
    :param do_polish_article: If True, polish the article by adding a summarization section and (optionally) removing
                                duplicate content.
    :param remove_duplicate: If True, remove duplicate content from the article, during polishing.
    """

    lm_configs = STORMWikiLMConfigs()

    vllm_kwargs = {
        "model": inference_model,
        "port": inference_port,
        "url": inference_url,
        "api_key": os.getenv("INFERENCE_API_KEY"),
        "stop": ('\n\n---',)  # dspy uses "\n\n---" to separate examples. Open models sometimes generate this.
    }

    conv_simulator_lm = VLLMClient(max_tokens=500, **vllm_kwargs)
    question_asker_lm = VLLMClient(max_tokens=500, **vllm_kwargs)
    outline_gen_lm = VLLMClient(max_tokens=400, **vllm_kwargs)
    article_gen_lm = VLLMClient(max_tokens=700, **vllm_kwargs)
    article_polish_lm = VLLMClient(max_tokens=4000, **vllm_kwargs)

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=output_dir,
        max_conv_turn=max_conv_turn,
        max_perspective=max_perspective,
        search_top_k=search_top_k,
        max_thread_num=max_thread_num,
        retrieve_top_k=retrieve_top_k
    )

    # STORM is a knowledge curation system which consumes information from the retrieval module.
    # Currently, the information source is the Internet and we use search engine API as the retrieval module.
    match retriever:
        case 'bing':
            rm = BingSearch(bing_search_api=os.getenv('SEARCH_API_KEY'), k=engine_args.search_top_k)
        case 'you':
             rm = YouRM(ydc_api_key=os.getenv('SEARCH_API_KEY'), k=engine_args.search_top_k)
        case 'brave':
            rm = BraveRM(brave_search_api_key=os.getenv('SEARCH_API_KEY'), k=engine_args.search_top_k)
        case 'duckduckgo':
            rm = DuckDuckGoSearchRM(k=engine_args.search_top_k, safe_search='On', region='us-en')
        case 'serper':
            rm = SerperRM(serper_search_api_key=os.getenv('SEARCH_API_KEY'), query_params={'autocorrect': True, 'num': 10, 'page': 1})
        case 'tavily':
            rm = TavilySearchRM(tavily_search_api_key=os.getenv('SEARCH_API_KEY'), k=engine_args.search_top_k, include_raw_content=True)
        case 'searxng':
            rm = SearXNG(searxng_api_key=os.getenv('SEARCH_API_KEY'), k=engine_args.search_top_k)
        case _:
             raise ValueError(f'Invalid retriever: {retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", or "searxng"')

    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    # Open LMs are generally weaker in following output format.
    # One way for mitigation is to add one-shot example to the prompt to exemplify the desired output format.
    # For example, we can add the following examples to the two prompts used in StormPersonaGenerator.
    # Note that the example should be an object of dspy.Example with fields matching the InputField
    # and OutputField in the prompt (i.e., dspy.Signature).
    find_related_topic_example = Example(
        topic="Knowledge Curation",
        related_topics="https://en.wikipedia.org/wiki/Knowledge_management\n"
                       "https://en.wikipedia.org/wiki/Information_science\n"
                       "https://en.wikipedia.org/wiki/Library_science\n"
    )
    gen_persona_example = Example(
        topic="Knowledge Curation",
        examples="Title: Knowledge management\n"
                 "Table of Contents: History\nResearch\n  Dimensions\n  Strategies\n  Motivations\nKM technologies"
                 "\nKnowledge barriers\nKnowledge retention\nKnowledge audit\nKnowledge protection\n"
                 "  Knowledge protection methods\n    Formal methods\n    Informal methods\n"
                 "  Balancing knowledge protection and knowledge sharing\n  Knowledge protection risks",
        personas="1. Historian of Knowledge Systems: This editor will focus on the history and evolution of knowledge curation. They will provide context on how knowledge curation has changed over time and its impact on modern practices.\n"
                 "2. Information Science Professional: With insights from 'Information science', this editor will explore the foundational theories, definitions, and philosophy that underpin knowledge curation\n"
                 "3. Digital Librarian: This editor will delve into the specifics of how digital libraries operate, including software, metadata, digital preservation.\n"
                 "4. Technical expert: This editor will focus on the technical aspects of knowledge curation, such as common features of content management systems.\n"
                 "5. Museum Curator: The museum curator will contribute expertise on the curation of physical items and the transition of these practices into the digital realm."
    )
    runner.storm_knowledge_curation_module.persona_generator.create_writer_with_persona.find_related_topic.demos = [
        find_related_topic_example]
    runner.storm_knowledge_curation_module.persona_generator.create_writer_with_persona.gen_persona.demos = [
        gen_persona_example]

    # A trade-off of adding one-shot example is that it will increase the input length of the prompt. Also, some
    # examples may be very long (e.g., an example for writing a section based on the given information), which may
    # confuse the model. For these cases, you can create a pseudo-example that is short and easy to understand to steer
    # the model's output format.
    # For example, we can add the following pseudo-examples to the prompt used in WritePageOutlineFromConv and
    # ConvToSection.
    write_page_outline_example = Example(
        topic="Example Topic",
        conv="Wikipedia Writer: ...\nExpert: ...\nWikipedia Writer: ...\nExpert: ...",
        old_outline="# Section 1\n## Subsection 1\n## Subsection 2\n"
                    "# Section 2\n## Subsection 1\n## Subsection 2\n"
                    "# Section 3",
        outline="# New Section 1\n## New Subsection 1\n## New Subsection 2\n"
                "# New Section 2\n"
                "# New Section 3\n## New Subsection 1\n## New Subsection 2\n## New Subsection 3"
    )
    runner.storm_outline_generation_module.write_outline.write_page_outline.demos = [write_page_outline_example]
    write_section_example = Example(
        info="[1]\nInformation in document 1\n[2]\nInformation in document 2\n[3]\nInformation in document 3",
        topic="Example Topic",
        section="Example Section",
        output="# Example Topic\n## Subsection 1\n"
               "This is an example sentence [1]. This is another example sentence [2][3].\n"
               "## Subsection 2\nThis is one more example sentence [1]."
    )
    runner.storm_article_generation.section_gen.write_section.demos = [write_section_example]

    runner.run(
        topic=topic,
        do_research=do_research,
        do_generate_outline=do_generate_outline,
        do_generate_article=do_generate_article,
        do_polish_article=do_polish_article,
        remove_duplicate=remove_duplicate
    )
    runner.post_run()
    runner.summary()


if __name__ == '__main__':
    parser = ArgumentParser()
    # global arguments
    parser.add_argument('--url', type=str, default='http://localhost',
                        help='URL of the VLLM server.')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port of the VLLM server.')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-70B-Instruct-fast',
                        help='Model to use for inference.')
    parser.add_argument('--output-dir', type=str, default='./results/output',
                        help='Directory to store the outputs.')
    parser.add_argument('--max-thread-num', type=int, default=3,
                        help='Maximum number of threads to use. The information seeking part and the article generation'
                             'part can speed up by using multiple threads. Consider reducing it if keep getting '
                             '"Exceed rate limit" error when calling LM API.')
    parser.add_argument('--retriever', type=str, choices=['bing', 'you', 'brave', 'serper', 'duckduckgo', 'tavily', 'searxng'],
                        help='The search engine API to use for retrieving information.')
    # stage of the pipeline
    parser.add_argument('--do-research', action='store_true',
                        help='If True, simulate conversation to research the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-outline', action='store_true',
                        help='If True, generate an outline for the topic; otherwise, load the results.')
    parser.add_argument('--do-generate-article', action='store_true',
                        help='If True, generate an article for the topic; otherwise, load the results.')
    parser.add_argument('--do-polish-article', action='store_true',
                        help='If True, polish the article by adding a summarization section and (optionally) removing '
                             'duplicate content.')
    # hyperparameters for the pre-writing stage
    parser.add_argument('--max-conv-turn', type=int, default=3,
                        help='Maximum number of questions in conversational question asking.')
    parser.add_argument('--max-perspective', type=int, default=3,
                        help='Maximum number of perspectives to consider in perspective-guided question asking.')
    parser.add_argument('--search-top-k', type=int, default=3,
                        help='Top k search results to consider for each search query.')
    # hyperparameters for the writing stage
    parser.add_argument('--retrieve-top-k', type=int, default=3,
                        help='Top k collected references for each section title.')
    parser.add_argument('--remove-duplicate', action='store_true',
                        help='If True, remove duplicate content from the article.')

    arguments = parser.parse_args()

    storm(topic="wireless capable audiophile headsets",
          inference_model=arguments.model,
          inference_url=arguments.url,
          inference_port=arguments.port,
          output_dir=arguments.output_dir,
          max_conv_turn=arguments.max_conv_turn,
          max_perspective=arguments.max_perspective,
          search_top_k=arguments.search_top_k,
          max_thread_num=arguments.max_thread_num,
          retriever=arguments.retriever,
          retrieve_top_k=arguments.retrieve_top_k,
          do_research=arguments.do_research,
          do_generate_outline=arguments.do_generate_outline,
          do_generate_article=arguments.do_generate_article,
          do_polish_article=arguments.do_polish_article,
          remove_duplicate=arguments.remove_duplicate
          )