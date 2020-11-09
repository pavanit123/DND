# Imports
import os
import subprocess

import azure.cognitiveservices.speech as speechsdk
import stanfordnlp

#stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', treebank='en_ewt', use_gpu=False,
                           pos_batch_size=3000)  # Build the pipeline, specify part-of-speech processor's batch size
def getSpeech():
    with open('keys/speech_key.txt', 'r') as f_open:
        speech_key = f_open.read()
        f_open.close()
    with open('keys/speech_region.txt', 'r') as f_open:
        service_region = f_open.read()
        f_open.close()
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Creates a recognizer with the given settings
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something to translate...")
    result = speech_recognizer.recognize_once()
    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}\n".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
        quit()
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
        quit()
    return result.text
def parse(text):
    # Process text input
    doc = nlp(text)  # Run the pipeline on text input
    for sentence in doc.sentences:
        translation = translate(sentence)
        result = []
        for word in translation[0]:
            result.append((word['text'].lower(), word['lemma'].lower()))
        print("\nResult: ", result, "\n")
        display(translation)
    return doc
def wordToDictionary(word):
    dictionary = {
        'index': word.index,
        'governor': word.governor,
        'text': word.text.lower(),
        'lemma': word.lemma.lower(),
        'upos': word.upos,
        'xpos': word.xpos,
        'dependency_relation': word.dependency_relation,
        'feats': word.dependency_relation,
        'children': []
    }
    return dictionary
def getMeta(sentence):
    # sentence.print_dependencies()
    englishStruct = {}
    aslStruct = {
        'rootElements': [],
        'UPOS': {
            'ADJ': [], 'ADP': [], 'ADV': [], 'AUX': [], 'CCONJ': [], 'DET': [], 'INTJ': [], 'NOUN': [], 'NUM': [],
            'PART': [], 'PRON': [], 'PROPN': [], 'PUNCT': [], 'SCONJ': [], 'SYM': [], 'VERB': [], 'X': []
        }
    }
    reordered = []
    # aslStruct["rootElements"] = []
    # Make a list of all tokenized words. This step might be unnecessary.
    words = []
    for token in sentence.tokens:
        # print(token)
        for word in token.words:

            print(word.index, word.governor, word.text, word.lemma, word.upos,
                  word.dependency_relation)  # , word.feats)
            # # Insert as dict
            # words.append(wordToDictionary(word))
            # Insertion sort
            j = len(words)
            for i, w in enumerate(words):
                if word.governor <= w['governor']:
                    continue
                else:
                    j = i
                    break
            # Convert to Python native structure when inserting.
            words.insert(j, wordToDictionary(word))
    # # Python sort for converted words
    # words.sort(key=attrgetter('governor', 'age')) # , reverse=True
    # words.sort(key=words.__getitem__) # , reverse=True
    reordered = words
    # print("\n", aslStruct, "\n")
    return reordered
def getLemmaSequence(meta):
    tone = ""
    translation = []
    for word in meta:
        # Remove blacklisted words
        if (word['text'].lower(), word['lemma'].lower()) not in (
        ('is', 'be'), ('the', 'the'), ('of', 'the'), ('is', 'are'), ('by', 'by'), (',', ','), (';', ';'), (':'), (':')):

            # Get Tone: get the sentence's tone from the punctuation
            if word['upos'] == 'PUNCT':
                if word['lemma'] == "?":
                    tone = "?"
                elif word['lemma'] == "!":
                    tone = "!"
                else:
                    tone = ""
                continue

            # Remove symbols and the unknown
            elif word['upos'] == 'SYM' or word['upos'] == 'X':
                continue

            # Remove particles
            if word['upos'] == 'PART':
                continue

            # Convert proper nouns to finger spell
            elif word['upos'] == 'PROPN':
                fingerSpell = []
                for letter in word['text'].lower():
                    print(letter)
                    spell = {}
                    spell['text'] = letter
                    spell['lemma'] = letter
                    # Add fingerspell as individual lemmas
                    fingerSpell.append(spell)
                print(fingerSpell)
                translation.extend(fingerSpell)
                print(translation)

            # Numbers
            elif word['upos'] == 'NUM':
                fingerSpell = []
                for letter in word['text'].lower():
                    spell = {}
                    # Convert number to fingerspell
                    pass
                    # Add fingerspell as individual lemmas
                    fingerSpell.append(spell)

            # Interjections usually use alternative or special set of signs
            elif word['upos'] == 'CCONJ':
                translation.append(word)

            # Interjections usually use alternative or special set of signs
            elif word['upos'] == 'SCONJ':
                if (word['text'].lower(), word['lemma'].lower() not in (('that', 'that'))):
                    translation.append(word)

            # Interjections usually use alternative or special set of signs
            elif word['upos'] == 'INTJ':
                translation.append(word)

            # Adpositions could modify nouns
            elif word['upos'] == 'ADP':
                # translation.append(word)
                pass

            # Determinants modify noun intensity
            elif word['upos'] == 'DET':
                pass

            # Adjectives modify nouns and verbs
            elif word['upos'] == 'ADJ':
                translation.append(word)
                # pass

            # Pronouns
            elif word['upos'] == 'PRON' and word['dependency_relation'] not in ('nsubj'):
                translation.append(word)

            # Nouns
            elif word['upos'] == 'NOUN':
                translation.append(word)

            # Adverbs modify verbs, leave for wh questions
            elif word['upos'] == 'ADV':
                translation.append(word)

            elif word['upos'] == 'AUX':
                pass

            # Verbs
            elif word['upos'] == 'VERB':
                translation.append(word)

    # translation = tree
    return (translation, tone)
def translate(parse):
    meta = getMeta(parse)
    translation = getLemmaSequence(meta)
    return translation
def display(translation):
    folder = os.getcwd()
    filePrefix = folder + "/videos/"
    # Alter ASL lemmas to match sign's file names.
    # In production, these paths would be stored at the dictionary's database.
    files = [filePrefix + word['text'].lower() + "_.mp4" for word in translation[0]]
    # Run video sequence using the MLT Multimedia Framework
    print("Running command: ", ["melt"] + files)
    process = subprocess.Popen(["melt"] + files + [filePrefix + "black.mp4"], stdout=subprocess.PIPE)
    result = process.communicate()
def main():
    flag = False
    while not flag:

        tests = [
            # "Where is the bathroom?",
            # "What is your name?",
            # "I'm Javier.",
            # "My name is Javier.",
            # "Bring your computer!",
            # "It's lunchtime!",
            # "Small dogs are cute",
            # "Chihuahuas are cute because they're small."
        ]

        if len(tests) == 0:
            tests = tests + [getSpeech()]

        if len(tests[0]) == 0:
            print("No speech detected... Reattempting.")
        else:
            for text in tests:
                print("Text to process: ", text, "\n")

                parse(text)

                print('\nPress "Enter" to continue or any type anything else to exit.')
                key = input()
                if key != '':
                    flag = True


main()