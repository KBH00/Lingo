"""
If context length is 0, return empty string to not consider the context length.

If sentence is True, Return cntxt_len - 1 number of sentences
                         before and after of the target word.
    If cntxt_len is 1, Return the sentence including target word.

If sentence is False, Return cntxt_len number of words
                         before and after of the target word.

"""

def get_context(target_word, context, sentence, cntxt_len) -> str:

    # if cntxt_len == 0:
    #     return ""
    
    import re
    sentences = re.split(r'[.!?] ', context)

    if cntxt_len == 1:
        target_sentence = None
        for sent in sentences:
            if target_word.lower() in sent.lower():
                target_sentence = sent
                return target_sentence

    if target_sentence is None:
        return "Target word not found in any sentence."

    sent_index = sentences.index(target_sentence)

    if sentence:
        start_index = max(0, sent_index - cntxt_len)
        end_index = min(len(sentences), sent_index + cntxt_len + 1)
        context = ". ".join(sentences[start_index:end_index])
    else:
        # words = target_sentence.split()
        words = context.split()
        word_index = words.index(target_word)
        start_index = max(0, word_index - cntxt_len)
        end_index = min(len(words), word_index + cntxt_len + 1)
        context = " ".join(words[start_index:end_index])

    return context