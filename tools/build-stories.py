"""Build corpus-stories.js from the TinyStories dataset.

TinyStories (Eldan & Li, 2023) is a set of very short stories written with a small
vocabulary: https://huggingface.co/datasets/roneneldan/TinyStories
It is licensed under CDLA-Sharing-1.0, and so is the sample this script writes.

Usage:
    1. Download TinyStoriesV2-GPT4-valid.txt from the dataset page.
    2. python tools/build-stories.py path/to/TinyStoriesV2-GPT4-valid.txt

Options let you trade page weight for knowledge: more sentences and a larger
vocabulary make the model know more, and the page slower to load.
"""
import argparse
import collections
import io
import os
import re

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('source', help='TinyStories text file')
ap.add_argument('--sentences', type=int, default=30000, help='how many sentences to keep')
ap.add_argument('--vocab', type=int, default=5000, help='keep only sentences made of the N most common words')
ap.add_argument('--min-words', type=int, default=4)
ap.add_argument('--max-words', type=int, default=14)
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'corpus-stories.js'))
args = ap.parse_args()

WORD = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
text = io.open(args.source, encoding='utf-8', errors='replace').read()
text = text.replace('’', "'").replace('‘', "'")

stories = []   # each story is a list of sentences, each sentence a list of words
for story in text.split('<|endoftext|>'):
    sents = []
    for raw in re.split(r'[.!?\n]+', story):
        words = [w.strip("'") for w in re.sub(r"[^a-z' ]", ' ', raw.lower()).split()]
        words = [w for w in words if w]
        if args.min_words <= len(words) <= args.max_words and all(WORD.match(w) for w in words):
            sents.append(words)
    if sents:
        stories.append(sents)

freq = collections.Counter(w for st in stories for s in st for w in s)
keep = {w for w, _ in freq.most_common(args.vocab)}

picked, seen = [], set()
for st in stories:
    for s in st:
        line = ' '.join(s)
        if line in seen or not all(w in keep for w in s):
            continue
        seen.add(line)
        picked.append(line)
    if len(picked) >= args.sentences:
        break
picked = picked[:args.sentences]

words = {w for line in picked for w in line.split()}
header = """    /* Short story sentences sampled from the TinyStories dataset (Eldan & Li, 2023):
       https://huggingface.co/datasets/roneneldan/TinyStories
       The dataset, and this sample of it, are licensed under CDLA-Sharing-1.0
       (https://cdla.dev/sharing-1-0/). Written by tools/build-stories.py; do not edit by hand.
       %d sentences, %d different words. corpus.js adds them to the training text. */
""" % (len(picked), len(words))
with io.open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(header + '    const STORY_TEXT = `' + '\n'.join(picked) + '`;\n')
print('%d sentences, %d different words, %.2f MB -> %s' % (len(picked), len(words), os.path.getsize(args.out) / 1e6, os.path.normpath(args.out)))
