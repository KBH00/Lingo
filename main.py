import argparse
from abbreviation.non_ai import *
from engine import synonym_recommendation

def get_args_parser():
    parser = argparse.ArgumentParser('Set Synonym recommendation', add_help=False)
    parser.add_argument('--model', default='llama2', type=str, help="name of model to inference")
    parser.add_argument('--target_word', default='CT', type=str, help='word to find synonym')

    parser.add_argument('--sentence', default=True, type=bool, help='Unit for context consideration')
    parser.add_argument('--context_len', default=1, type=int, help='Number of unit')

    parser.add_argument('--text', default='The radiological evaluation of patients with acute spinal trauma has always been a challenging problem. Multiple radiological procedures are often necessary for \
    #                                            complete evaluation of the extent of spinal injury. CT provides an ideal modality whereby accurate assessment of displacement of bony fragments as well as associated spinal cord and nerve root injury can easily be performed, \
    #                                            eliminating the need for difficult radiological procedures.', type=str, help='')
    parser.add_argument('--abbrv', default=False, type=bool, help='Turn on or off to change abbreviation')
    return parser

# abbreviations = load_abbreviations('abbreviation.txt')
# input_sentence = "The patient's C1 and C2 vertebrae are aligned, and BP, CT, CHF is stable."

# output_sentence = replace_abbreviations(input_sentence, abbreviations)

def main(args):
    # synonym_recommend = synonym_recommendation(model="llama2", targetWord="CT", sentence=True, cntxt_len=2, text="The radiological evaluation of patients with acute spinal trauma has always been a challenging problem. Multiple radiological procedures are often necessary for \
    #                                            complete evaluation of the extent of spinal injury. CT provides an ideal modality whereby accurate assessment of displacement of bony fragments as well as associated spinal cord and nerve root injury can easily be performed, \
    #                                            eliminating the need for difficult radiological procedures.")

    if args.abbrv == True:
        abbreviations = load_abbreviations('abbreviation.txt')
        output_abb = replace_abbreviations(args.text, abbreviations)
    
    synonym_recommend = synonym_recommendation(model=args.model, targetWord=args.target_word, sentence=args.sentence, 
                                               cntxt_len=args.context_len, text=args.text)
    print(synonym_recommend.post_processing())


if __name__=='__main__':
    parser = argparse.ArgumentParser('Synonym recommendation option script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)