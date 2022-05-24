
# QA Machine Learning Model
QA (Question-Answering) are machine or deep learning models that 
can answer questions given some context, and sometimes without any context
based on the data already fed into the model. They can extract 
answer phrases from paragraphs and can even paraphrase the answer 
generatively.

### What Is a QA Pipeline ?
QA or Question-Answering Model uses Hugging Face pre-trained
Transformers for predicting the answers from the context provided
to the model. These models are trained on a large dataset and 
have high accuracy. For more details and model have a look at the
following link:

https://huggingface.co/models?pipeline_tag=question-answering&sort=downloads

### How does QA pipeline (project) works?
QA Pipeline uses Hugging Face pre-trained Transformers model for 
predicting the answers from the context provided. In this project
we wil be using 'bert-large-uncased-whole-word-masking-finetuned-squad'
pre-trained model and tokenizer on question-answering pipeline. 
The model takes context and questions as the input and returns a list
of outputs.

#### Structure of the Project:

    Input_Sample
    | (Text File Sample for testing)
    Jupyter Notebook
    | QnA.ipynb
    Pre-Trained Model
        QA
            | (Pre-Trained Model Files)
        tokenizer
            | (Tokenizer files)
    README.md

#### Installation of Packages:
Install packages using following commands:

    !pip tensorflow
    !pip install  transformers
    !pip install  re

#### Tokenizer and Model Loading:
After loading the necessary packages, import the packages into the 
notebook and load the tokenizer and model using following code:

    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from transformers import pipeline
    
    class QA:
        def __init__(self):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
                self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
                self.qna = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)
                print("Model Loaded")
            except:
            print("Error Loading Model")

The model is loaded into a class. If the code prints "Error Loading Model"
check for the model name.

#### Input Text Cleaning:
This function takes string as input and removes all the irregularities
in the string like the extract space, punctuations and special characters.

    def cleaning_text(self,text:str)->str:
        # punc = string.punctation
        # nltk.corpus.stopwords.words("english")

        text = text.lstrip().rstrip()
        self.temp = re.sub("\n+"," ",text)

        return self.temp

This function is build in the QA class.

#### Input Query to the Model:
The input in the model is of following type:

    question = [["What is present....?"],
            ["Of the two countries...?"]]
            
    context = ["This combination of cancellations and σ and π overlaps results in dioxygen’s double bond character and reactivity, and a triplet electronic 
    ground........","
    Between 1991 and 2000, the total area of forest lost in the Amazon rose from 415,000 to 587,000 square kilometres (160,000 to 227,000 sq mi), with most
    of the lost forest.... "]

The questions corresponding to the context are given in the list
format.

#### Predicting the Answers:
The following function in the QA class takes context, and questions
as input in the above described format and returns a list containing
dictionaries .

    def predict_sol(self,context,ques):
        ans = []

        try:
            if str(type(ques)) == "<class 'list'>":

                for i in ques:
                req_ques = {
                    'question':i,
                    'context':context
                }
                sol = { 'ques':i, 'ans':self.qna(req_ques)['answer'] }

                ans.append(sol)

            else:
                req_ques = {
                    'question':ques,
                    'context':context
                }
                sol = {
                    'ques':ques,
                    'ans':self.qna(req_ques)['answer']
                }
                ans.append(sol)
            return ans      
        
        except:
          pass

#### Output Format:
A list containing list of dictionaries as output.

    [[{'ans': 'electrons',
    'ques': 'What is present....?'}],
    [{'ans': 'Brazil',
    'ques': 'Of the two countries...?'}]]
## Authors

- [@Rachit R Jindal](https://www.github.com/rachitjindal56)


## Tech Stack

**Server:** Python, Transformers, Hugging Face, 

