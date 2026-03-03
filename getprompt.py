import json


def get_search_prompt(query, tree_without_text):
    search_prompt = f"""
        You are given a question and a tree structure of a document.
        Each node contains a node id, node title, and a corresponding summary.
        Your task is to find all nodes that are likely to contain the answer to the question.

        Question: {query}

        Document tree structure:
        {json.dumps(tree_without_text, indent=2)}

        Please reply in the following JSON format:
        {{
            "thinking": "<Your thinking process on which nodes are relevant to the question>",
            "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
        }}
        Directly return the final JSON structure. Do not output anything else.
    """
    return search_prompt
