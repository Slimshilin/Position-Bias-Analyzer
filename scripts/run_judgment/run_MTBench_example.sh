python3 subeval/subjective/sub_eval.py --data QA_data/example/MTBench_data_sliced.xlsx --benchmark MTBench --model gpt-4 claude-v1 --refm vicuna-13b-v1.3 --judge gemini-pro --eval-nopt 3 --eval-proc 1 --mode dual

python3 subeval/subjective/sub_eval.py --data QA_data/example/MTBench_data_sliced.xlsx --benchmark MTBench --model gpt-4 claude-v1 --refm vicuna-13b-v1.3 --judge claude-3-haiku-20240307  --eval-nopt 3 --eval-proc 1 --mode dual

python3 subeval/subjective/sub_eval.py --data QA_data/example/MTBench_data_sliced.xlsx --benchmark MTBench --model gpt-4 claude-v1 --refm vicuna-13b-v1.3 --judge gpt-3.5-turbo-1106  --eval-nopt 3 --eval-proc 1 --mode dual