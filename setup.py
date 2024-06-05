from setuptools import find_packages, setup

REQUIRES = """
anthropic==0.25.8
google-generativeai==0.5.2
matplotlib==3.8.4
numpy==1.26.4
openai==1.27.0
openpyxl==3.1.2
pandas==2.2.2
pillow==10.3.0
portalocker==2.8.2
requests==2.31.0
rich==13.7.1
scikit-learn==1.4.2
scipy==1.13.0
seaborn==0.13.2
statsmodels==0.14.2
tabulate==0.9.0
tiktoken==0.4.0
timeout-decorator==0.5.0
tqdm==4.66.4
"""

def get_install_requires():
    reqs = [req for req in REQUIRES.split('\n') if len(req) > 0]
    return reqs

with open('README.md') as f:
    readme = f.read()

def do_setup():
    setup(
        name='subeval',
        version='0.1.0',
        description='Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs',
        author="Lin Shi",
        long_description=readme,
        long_description_content_type='text/markdown',
        cmdclass={},
        install_requires=get_install_requires(),
        setup_requires=[],
        python_requires='>=3.7.0',
        packages=find_packages(exclude=[
            'test*',
            'paper_test*',
        ]),
        keywords=['AI', 'NLP', 'Position Bias', 'Judging the Judges', 'LLM-as-a-Judge', 'Fairness'],
        entry_points={
            "console_scripts": [
                "sub_eval = subeval.subjective.sub_eval:subeval_call", 
            ]
        },
        classifiers=[
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
        ]
    )

if __name__ == '__main__':
    do_setup()
