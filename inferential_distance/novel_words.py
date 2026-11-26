"""Step 1: mark the hypothesis words that are novel with respect to the premise.

A hypothesis word is *novel* when it is not a stopword, is purely alphabetic
ASCII, and the Lancaster stem of its spaCy lemma shares no substring relation
with the stem of any (non-stopword, alphabetic, ASCII) premise word.

Novel words are highlighted in the hypothesis by wrapping them in ``**``, and the
highlighted hypothesis is what the atom-extraction prompt receives.

This module reproduces ``get_novel_words`` and the highlighting loop of the
original research code without changes in behaviour (see README, "Known
quirks").
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_SPACY_MODEL = "en_core_web_trf"

# Stopword list used by the original code (copied verbatim, order preserved).
STOPWORDS = "0,1,2,3,4,5,6,7,8,9,a,A,about,above,across,after,again,against,all,almost,alone,along,already,also,although,always,am,among,an,and,another,any,anyone,anything,anywhere,are,aren't,around,as,at,b,B,back,be,became,because,become,becomes,been,before,behind,being,below,between,both,but,by,c,C,can,cannot,can't,could,couldn't,d,D,did,didn't,do,does,doesn't,doing,done,don't,down,during,e,E,each,either,enough,even,ever,every,everyone,everything,everywhere,f,F,few,find,first,for,four,from,full,further,g,G,get,give,go,h,H,had,hadn't,has,hasn't,have,haven't,having,he,he'd,he'll,her,here,here's,hers,herself,he's,him,himself,his,how,however,how's,i,I,i'd,if,i'll,i'm,in,interest,into,is,isn't,it,it's,its,itself,i've,j,J,k,K,keep,l,L,last,least,less,let's,m,M,made,many,may,me,might,more,most,mostly,much,must,mustn't,my,myself,n,N,never,next,no,nobody,noone,nor,not,nothing,now,nowhere,o,O,of,off,often,on,once,one,only,or,other,others,ought,our,ours,ourselves,out,over,own,p,P,part,per,perhaps,put,q,Q,r,R,rather,s,S,same,see,seem,seemed,seeming,seems,several,shan't,she,she'd,she'll,she's,should,shouldn't,show,side,since,so,some,someone,something,somewhere,still,such,t,T,take,than,that,that's,the,their,theirs,them,themselves,then,there,therefore,there's,these,they,they'd,they'll,they're,they've,this,those,though,three,through,thus,to,together,too,toward,two,u,U,under,until,up,upon,us,v,V,very,w,W,was,wasn't,we,we'd,we'll,well,we're,were,weren't,we've,what,what's,when,when's,where,where's,whether,which,while,who,whole,whom,who's,whose,why,why's,will,with,within,without,won't,would,wouldn't,x,X,y,Y,yet,you,you'd,you'll,your,you're,yours,yourself,yourselves,you've,z,Z,non".split(',')

HIGHLIGHT = "**"


@lru_cache(maxsize=None)
def load_spacy(model_name: str = DEFAULT_SPACY_MODEL):
    """Load (once) and return the spaCy pipeline used for lemmatisation."""
    import spacy

    return spacy.load(model_name)


@lru_cache(maxsize=1)
def _lancaster():
    from nltk.stem import LancasterStemmer

    return LancasterStemmer()


def stemmer(word: str) -> str:
    return _lancaster().stem(word)


def contains_only_english_characters(s: str) -> bool:
    return all(ord(char) <= 127 for char in s)


def stem_and_lemmatize_sentence(s: str, nlp=None):
    """Return ``(triples, tokens)`` where each triple is (stem of lemma, word, token).

    Stopwords, non-alphabetic tokens and tokens with non-ASCII characters are skipped.
    """
    nlp = nlp if nlp is not None else load_spacy()
    doc = nlp(s)
    lemma_and_stem = []
    originals = []
    for d in doc:
        if d.text.lower() not in STOPWORDS and d.text.isalpha() and contains_only_english_characters(d.text):
            lemma_and_stem.append((stemmer(d.lemma_), d.text, d))
            originals.append(d)
    return lemma_and_stem, originals


def remove_overlapping_strings(s_list1, s_list2):
    """Words of ``s_list2`` whose stem is not a substring of, and does not contain,
    any stem in ``s_list1``."""
    result = []
    stems1 = [s[0] for s in s_list1]
    for stem2, w2, d2 in s_list2:
        if all(stem1 not in stem2 and stem2 not in stem1 for stem1 in stems1):
            result.append(w2)
    return result


def get_novel_words(premise: str, hypothesis: str, nlp=None) -> list[str]:
    """Hypothesis words (surface form, in order, duplicates kept) that are novel
    with respect to the premise."""
    s1_stems, _ = stem_and_lemmatize_sentence(premise, nlp)
    s2_stems, _ = stem_and_lemmatize_sentence(hypothesis, nlp)
    return remove_overlapping_strings(s1_stems, s2_stems)


def highlight(hypothesis: str, novel_words: list[str]) -> str:
    """Wrap every occurrence of every novel word in ``**`` (plain substring
    replacement, applied word by word, exactly as in the original code)."""
    h_highlighted = hypothesis
    for w in novel_words:
        h_highlighted = h_highlighted.replace(w, HIGHLIGHT + w + HIGHLIGHT)
    return h_highlighted


def highlight_novel_words(premise: str, hypothesis: str, nlp=None) -> str:
    return highlight(hypothesis, get_novel_words(premise, hypothesis, nlp))
