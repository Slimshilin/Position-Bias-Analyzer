from subeval import *
import numpy as np

# Columns of the data Dataframe
MTBench_RESERVE_KEYS = ['question_1', 'question_2', 'index', 'evaluating_guidance', 'task', 'reference_answer_1', 'reference_answer_2']
DevBench_RESERVE_KEYS = ['question', 'index', 'source', 'evaluating_guidance', 'task', 'reference_answer']

def json2tsv(json_name, benchmark):
    data = load(json_name)
    res = defaultdict(list)
    keys_all = []
    for item in data:
        keys_all.extend(list(item.keys()))
    keys_all = set(keys_all)

    if benchmark == "MTBench":
        for _, item in enumerate(data):
            for k in keys_all:
                if k in MTBench_RESERVE_KEYS:
                    res[k].append(item[k] if k in item else None)
                elif k.startswith('answer_1-') or k.startswith('answer_2-'):
                    res[k].append(item[k] if k in item else None)
                else:
                    res['answer_1-' + k].append(item[k] if k in item else None)
                    res['answer_2-' + k].append(item[k] if k in item else None)

    elif benchmark == "DevBench":
        for _, item in enumerate(data):
            for k in keys_all:
                if k in DevBench_RESERVE_KEYS:
                    res[k].append(item[k] if k in item else None)
                elif k.startswith('answer-'):
                    res[k].append(item[k] if k in item else None)
                else:
                    res['answer-' + k].append(item[k] if k in item else None)

    res = pd.DataFrame(res)
    dump(res, json_name.replace('.json', '.tsv'), quoting=csv.QUOTE_ALL)
    return json_name.replace('.json', '.tsv') 

# Function to generate inference input for a subjective evaluation task
def generate_inference_input(meta_file, benchmark, model=None, refm=None, mode='random', fill_contents=[], seed=2680):
    '''
    Generate data for comparison
    Params:
        model: the model name for evaluation
        refm: the reference model for evaluation comparison
        mode: choose from [random,dual]
            random: randomly choose between 'normal' (reference;model) and 'rev' (model;reference)
            dual: Stacking the data for 'normal' and 'rev' together
        seed:
            positive: no change with refm and model
            negative: swapping refm and model
    '''
    flip = False  # Flag to indicate if the order of models should be flipped
    if seed < 0:  # Flipping the order if the seed is negative
        seed = -seed
        flip = True

    np.random.seed(seed)  # Setting the random seed

    # Loading the meta file
    meta = load(meta_file)  
    for col in fill_contents:
        load_file_content(meta, col)

    # Creating a map from index to task if 'task' column is present
    task_map = {x: y for x, y in zip(meta['index'], meta['task'])} if 'task' in meta else None

    ### Handles different Benchmarks differently ###
    if benchmark == "MTBench": # 2-round QA
        # Extracting relevant columns from the meta file
        questions_1 = meta['question_1']
        questions_2 = meta['question_2']
        index = meta['index']
        refanswer_1 = meta['reference_answer_1']
        refanswer_2 = meta['reference_answer_2']
        guidance = meta['evaluating_guidance']
        num_q = len(index)  # Number of questions

        # Mapping index to questions, reference answers, and guidance
        index_map = {
            index[i]: dict(question_1=questions_1[i], 
                        question_2=questions_2[i],
                        refanswer_1=refanswer_1[i],
                        refanswer_2=refanswer_2[i],
                        guidance=guidance[i]) for i in range(num_q)}

        # Initializing lists to store comparison data
        '''
        Q1: question_1
        Q2: question_2
        A: First model name
        B: Second model name
        A1_1: Answer 1 from model 1
        A1_2: Answer 2 from model 1
        A2_1: Answer 1 from model 2
        A2_2: Answer 2 from model 2
        CI: Comparison Index
        REFA1: Reference answer 1
        REFA2: Reference answer 2
        EVALG: Evaluation guidance
        '''
        Q1, Q2, A, B, A1_1, A1_2, A2_1, A2_2, CI = [], [], [], [], [], [], [], [], []
        REFA1, REFA2, EVALG = [], [], []

        # Extracting model names from the meta file
        meta_keys = list(meta.keys())
        models = [x[9:] for x in meta_keys if x.startswith('answer_1-')]

        # Ensuring the answers are strings
        for m in models:
            answers_1 = meta[f'answer_1-{m}']
            answers_2 = meta[f'answer_2-{m}']
            answers_1 = [str(x) for x in answers_1]
            answers_2 = [str(x) for x in answers_2]
            meta[f'answer_1-{m}'] = answers_1
            meta[f'answer_2-{m}'] = answers_2

        # Checking if the reference model and comparison model are provided and valid
        assert refm is not None and refm in models
        assert model is not None and model in models

        if flip:  # Swapping the models if flip flag is set
            model, refm = refm, model

        # Generating comparison data for each question
        for i in range(num_q):
            item = meta.iloc[i] # each question row
            qidx = item['index'] # question index

            # Skipping if answers from either model are missing
            if pd.isna(item[f'answer_1-{model}']) or pd.isna(item[f'answer_2-{model}']) or \
            pd.isna(item[f'answer_1-{refm}']) or pd.isna(item[f'answer_2-{refm}']):
                continue

            curmode = mode
            # Randomly choosing the order of models (normal or reverse order) for comparison if mode is 'random'
            if curmode == 'random':
                curmode = 'normal' if np.random.random() < 0.5 else 'rev'

            # Appending data for normal and dual modes
            # A as reference and B as model
            if curmode in ['normal', 'dual']:
                CI.append(';'.join([str(qidx), refm, model]))
                A.append(refm)
                B.append(model)
                Q1.append(index_map[qidx]['question_1'].strip())
                Q2.append(index_map[qidx]['question_2'].strip())
                REFA1.append(index_map[qidx]['refanswer_1'])
                REFA2.append(index_map[qidx]['refanswer_2'])
                EVALG.append(index_map[qidx]['guidance'])
                A1_1.append(item[f'answer_1-{refm}'].strip())
                A1_2.append(item[f'answer_2-{refm}'].strip())
                A2_1.append(item[f'answer_1-{model}'].strip())
                A2_2.append(item[f'answer_2-{model}'].strip())

            # Appending data for reverse and dual modes
            # A as model and B as reference
            if curmode in ['rev', 'dual']:
                CI.append(';'.join([str(qidx), model, refm]))
                B.append(refm)
                A.append(model)
                Q1.append(index_map[qidx]['question_1'].strip())
                Q2.append(index_map[qidx]['question_2'].strip())
                REFA1.append(index_map[qidx]['refanswer_1'])
                REFA2.append(index_map[qidx]['refanswer_2'])
                EVALG.append(index_map[qidx]['guidance'])
                A2_1.append(item[f'answer_1-{refm}'].strip())
                A2_2.append(item[f'answer_2-{refm}'].strip())
                A1_1.append(item[f'answer_1-{model}'].strip())
                A1_2.append(item[f'answer_2-{model}'].strip())

        # Creating a DataFrame from the comparison data
        data = pd.DataFrame(dict(cmp_index=CI, question_1=Q1, question_2=Q2, answer1_1=A1_1, answer1_2=A1_2, 
                                answer2_1=A2_1, answer2_2=A2_2, A=A, B=B, 
                                reference_answer_1=REFA1, reference_answer_2=REFA2, evaluating_guidance=EVALG))
    
    elif benchmark == "DevBench": # 1-round QA
        # Extracting relevant columns from the meta file
        questions = meta['question']
        index = meta['index']
        refanswer = meta['reference_answer']
        guidance = meta['evaluating_guidance']
        num_q = len(index)  # Number of questions

        # Mapping index to question, reference answer, and guidance
        index_map = {
            index[i]: dict(question=questions[i], 
                        refanswer=refanswer[i],
                        guidance=guidance[i]) for i in range(num_q)}

        # Initializing lists to store comparison data
        '''
        Q: question
        A: First model name
        B: Second model name
        A1: Answer from model 1
        A2: Answer from model 2
        CI: Comparison Index
        REFA: Reference answer
        EVALG: Evaluation guidance
        '''
        Q, A, B, A1, A2, CI = [], [], [], [], [], []
        REFA, EVALG = [], []

        # Extracting model names from the meta file
        meta_keys = list(meta.keys())
        models = [x[7:] for x in meta_keys if x.startswith('answer-')]

        # Ensuring the answers are strings
        for m in models:
            answers = meta[f'answer-{m}']
            answers = [str(x) for x in answers]
            meta[f'answer-{m}'] = answers

        # Checking if the reference model and comparison model are provided and valid
        assert refm is not None and refm in models
        assert model is not None and model in models

        if flip:  # Swapping the models if flip flag is set
            model, refm = refm, model

        # Generating comparison data for each question
        for i in range(num_q):
            item = meta.iloc[i] # each question row
            qidx = item['index'] # question index

            # Skipping if answers from either model are missing
            if pd.isna(item[f'answer-{model}']) or pd.isna(item[f'answer-{refm}']): continue

            curmode = mode
            # Randomly choosing the order of models (normal or reverse order) for comparison if mode is 'random'
            if curmode == 'random':
                curmode = 'normal' if np.random.random() < 0.5 else 'rev'

            # Appending data for normal and dual modes
            # A as reference and B as model
            if curmode in ['normal', 'dual']:
                CI.append(';'.join([qidx, refm, model]))
                A.append(refm)
                B.append(model)
                Q.append(index_map[qidx]['question'].strip())
                REFA.append(index_map[qidx]['refanswer'])
                EVALG.append(index_map[qidx]['guidance'])
                A1.append(item[f'answer-{refm}'].strip())
                A2.append(item[f'answer-{model}'].strip())
            # Appending data for reverse and dual modes
            # A as model and B as reference
            if curmode in ['rev', 'dual']:
                CI.append(';'.join([qidx, model, refm]))
                B.append(refm)
                A.append(model)
                Q.append(index_map[qidx]['question'].strip())
                REFA.append(index_map[qidx]['refanswer'])
                EVALG.append(index_map[qidx]['guidance'])
                A2.append(item[f'answer-{refm}'].strip())
                A1.append(item[f'answer-{model}'].strip())

        # Creating a DataFrame from the comparison data
        data = pd.DataFrame(dict(cmp_index=CI, question=Q, answer1=A1, answer2=A2, A=A, B=B, reference_answer=REFA, evaluating_guidance=EVALG))

    else:
        raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")

    # Adding the 'task' column if the task map is available
    if task_map is not None:
        data['task'] = [task_map[int(idx.split(';')[0])] for idx in data['cmp_index']]
    return data