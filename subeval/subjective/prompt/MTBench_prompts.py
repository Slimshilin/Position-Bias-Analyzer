'''
Prompts for MTBench

This is the prompt that's given to LLM Judges to generate the code.
Only considers the multi-run pairwise comparison cases.


The prompts for "math" tasks are different than others (named "general").
For "math" tasks:
- system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. Your evaluation should consider correctness and helpfulness. You will be given reference answers, the assistant A's answers, the assistant B's answers. Your job is to determine which assistant provides correct and helpful answers to the second user question. Begin your evaluation by comparing both assistants' answers with the reference answers. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

For other "general" tasks:
- system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides a better answer to the second user question. Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."


These system_prompts act as a prefix of the prompts give to LLM judges. The remaining part should follow this template:

For "math" tasks:
prompt_template = "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>"

For "general" tasks:
prompt_template = "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>"

where you should write a function `build_prompt(item)` that takes a row of the data file that includes the columns:
- question_1: The first question.
- question_2: The second question.
- answer1_1: The first answer from model 1.
- answer1_2: The second answer from model 1.
- answer2_1: The first answer from model 2.
- answer2_2: The second answer from model 2.
- A: The name of model 1.
- B: The name of model 2.
- cmp_index: The comparison index, which is a unique identifier for each comparison.
- reference_answer_1 (optional): The first reference answer, if available.
- reference_answer_2 (optional): The second reference answer, if available.
- evaluating_guidance (optional): The evaluation guidance, if available.
- task (optional): The task category, if available.

and then filled the prompt template accordingly. Remember in the prompt template it uses a and b for each response-generation model, whereas in the data file it uses 1 and 2 (and since this may cause confusion with answer 1 and answer 2, this requires extra attentaion to handle it correctly).
'''


def build_prompt(item):
    question_1 = item['question_1']
    question_2 = item['question_2']
    answer_a_1 = item['answer1_1']
    answer_a_2 = item['answer1_2']
    answer_b_1 = item['answer2_1']
    answer_b_2 = item['answer2_2']
    model_a = item['A']
    model_b = item['B']
    ref_answer_1 = item.get('reference_answer_1', '')
    ref_answer_2 = item.get('reference_answer_2', '')

    if 'task' in item and item['task'] == 'math':
        system_prompt = (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. "
            "Your evaluation should consider correctness and helpfulness. You will be given reference answers, the assistant A's answers, "
            "the assistant B's answers. Your job is to determine which assistant provides correct and helpful answers to the second user question. "
            "Begin your evaluation by comparing both assistants' answers with the reference answers. Identify and correct any mistakes. "
            "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "
            "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. "
            "After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" "
            "if assistant B is better, and \"[[C]]\" for a tie."
        )
        prompt_template = (
            "<|The Start of Reference Answer|>\n\n"
            "### User:\n{question_1}\n\n"
            "### Reference answer:\n{ref_answer_1}\n\n"
            "### User:\n{question_2}\n\n"
            "### Reference answer:\n{ref_answer_2}\n\n"
            "<|The End of Reference Answer|>\n\n\n"
            "<|The Start of Assistant A's Conversation with User|>\n\n"
            "### User:\n{question_1}\n\n"
            "### Assistant A:\n{answer_a_1}\n\n"
            "### User:\n{question_2}\n\n"
            "### Assistant A:\n{answer_a_2}\n\n"
            "<|The End of Assistant A's Conversation with User|>\n\n\n"
            "<|The Start of Assistant B's Conversation with User|>\n\n"
            "### User:\n{question_1}\n\n"
            "### Assistant B:\n{answer_b_1}\n\n"
            "### User:\n{question_2}\n\n"
            "### Assistant B:\n{answer_b_2}\n\n"
            "<|The End of Assistant B's Conversation with User|>"
        )
    else:
        system_prompt = (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. "
            "You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider "
            "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides "
            "a better answer to the second user question. Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. "
            "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length "
            "of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, "
            "output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
        )
        prompt_template = (
            "<|The Start of Assistant A's Conversation with User|>\n\n"
            "### User:\n{question_1}\n\n"
            "### Assistant A:\n{answer_a_1}\n\n"
            "### User:\n{question_2}\n\n"
            "### Assistant A:\n{answer_a_2}\n\n"
            "<|The End of Assistant A's Conversation with User|>\n\n\n"
            "<|The Start of Assistant B's Conversation with User|>\n\n"
            "### User:\n{question_1}\n\n"
            "### Assistant B:\n{answer_b_1}\n\n"
            "### User:\n{question_2}\n\n"
            "### Assistant B:\n{answer_b_2}\n\n"
            "<|The End of Assistant B's Conversation with User|>"
        )

    prompt = system_prompt + "\n\n" + prompt_template.format(
        question_1=question_1,
        answer_a_1=answer_a_1,
        question_2=question_2,
        answer_a_2=answer_a_2,
        answer_b_1=answer_b_1,
        answer_b_2=answer_b_2,
        ref_answer_1=ref_answer_1,
        ref_answer_2=ref_answer_2
    )

    return prompt