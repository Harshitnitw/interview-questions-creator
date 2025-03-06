prompt_template = """
You are an expert at creating questions on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

-------
{text}
-------

Create 5 questions that will prepare the coders or programmers for their tests.
Make sure not to lose any important information. Only give the questions, each separated by a new line.

QUESTIONS:
"""

refine_template = ("""
    You are an expert at creating practice questions based on coding material and documentation.
    Your goal is to prepare a coder or programmer for their exam and coding tests.
    We have recieved some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones,
    (only if necessary) with some more context below.
    -------
    {text}
    -------
                   
    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original question.
    We want 5 questions in total. Only give the questions, each separated by a new line.
    QUESTIONS:
"""
)