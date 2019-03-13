"""

"""

from flask import Flask, render_template, request
from wtforms import Form, TextField, validators, SubmitField, DecimalField, IntegerField
from process_sentence import process_sentence
from finish_sentence import finish_sentence
# create app

app = Flask(__name__)

class SentenceAnalyzeForm(Form):
    # Input sentence
    inputseq = TextField("Enter a sentence to analyze", validators=[validators.InputRequired()])

    # Submit button
    submit = SubmitField("Analyze")

class SentenceCreationForm(Form):
    # Input sentence
    inputseq = TextField("Enter few words to finish as a senetnce", validators=[validators.InputRequired()])

    # Submit button
    submit = SubmitField("Generate")

# Home - Analyze Sentence page
@app.route("/", methods=['GET', 'POST'])
def home():
    """
    """
    # create form
    form = SentenceAnalyzeForm(request.form)

    # on form entry
    if request.method == 'POST' and form.validate():
        # Extract information
        input_sentence = request.form['inputseq']

        input_sentence_len = len(input_sentence)
        input_sentence_liklihood, corrected_sentence, corrected_sentence_liklihood = process_sentence("English", input_sentence, 4)
        html = ''
        html = html + header('RNN Analyzed Sentence ', color='black')
        html = html + box("Input: ") # + '<br/>'
        html = html + box('\"' + input_sentence + '\"') # + '<br/>'
        html = html + box("Liklihood = " + f'{input_sentence_liklihood:.1E}') + '<br/>'
        html = html + box('', "Corrected: ")
        html = html + box('', '\"' + corrected_sentence + '\"' ) # + '<br/>'
        html = html + box('', "Liklihood = " + f'{corrected_sentence_liklihood:.1E}') + '<br/>'
        return render_template('results.html', input=f'<div>{html}</div>')
    return render_template('index.html', form=form)

# Finish sentence page
@app.route("/finish", methods=['GET', 'POST'])
def generate_sentence():
    """
    """
    # create form
    form = SentenceCreationForm(request.form)

    # on form entry
    if request.method == 'POST' and form.validate():
        # Extract information
        input_sequence = request.form['inputseq']

        input_sequence_len = len(input_sequence)
        gen_sentences = finish_sentence("English", input_sequence, 12, 5)
        html = ''
        html = html + header('RNN Genered Sentences ', color='black')
        for gen_sentence in gen_sentences:
            html = html + box(input_sequence, gen_sentence[input_sequence_len+6:]) # + '<br/>'

        return render_template('results.html', input=f'<div>{html}</div>')
    return render_template('finish.html', form=form)

def header(text, color='black'):
    """Create an HTML header"""

    raw_html = f'<h1 style="margin-top:16px;color: {color}"><center>' + text + '</center></h1>'

    return raw_html

def box(text1, text2=''):
    """Create an HTML box of text"""
    raw_html = '<div align=left style="padding:8px;font-size:28px;margin-top:5px;margin-bottom:5px">' + text1 + '<span style="color:red">' + text2 + '</div>'

    return raw_html

if __name__ == "__main__":
    # Run app
    app.run(host="0.0.0.0", port=8911)
