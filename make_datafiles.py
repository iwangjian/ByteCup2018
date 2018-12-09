import os
import hashlib
import subprocess
import collections
import json
import tarfile
import io
import pickle as pkl
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

dm_single_close_quote = '\u2019'
dm_double_close_quote = '\u201d'

# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]

all_train_urls = "./bytecup/data/lists/all_train.txt"
all_val_urls = "./bytecup/data/lists/all_valid.txt"
all_test_urls = "./bytecup/data/lists/all_test.txt"

train_stories_dir = './bytecup/data/train'
valid_stories_dir = './bytecup/data/valid'
test_stories_dir = './bytecup/data/test'

train_tokenized_stories_dir = "./bytecup/tokenized/train_tokenized"
valid_tokenized_stories_dir = "./bytecup/tokenized/valid_tokenized"
test_tokenized_stories_dir = "./bytecup/tokenized/test_tokenized"
finished_files_dir = "./bytecup/finished_files"

VOCAB_SIZE = 200000


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using
       Stanford CoreNLP Tokenizer
    """
    print("Preparing to tokenize {} to {}...".format(stories_dir,
                                                     tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write(
                "{} \t {}\n".format(
                    os.path.join(stories_dir, s),
                    os.path.join(tokenized_stories_dir, s)
                )
            )
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing {} files in {} and saving in {}...".format(
        len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of
    # files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory {} contains {} files, but it "
            "should contain the same number as {} (which has {} files). Was"
            " there an error during tokenization?".format(
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig)
        )
    print("Successfully finished tokenizing {} to {}.\n".format(
        stories_dir, tokenized_stories_dir))


def read_story_file(text_file):
    with open(text_file, "r") as f:
        # sentences are separated by 2 newlines
        # single newlines might be image captions
        # so will be incomplete sentence
        lines = f.read().split('\n\n')
    return lines


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def get_art_abs(story_file):
    """ return as list of sentences"""
    lines = read_story_file(story_file)

    # Lowercase, truncated trailing spaces, and normalize spaces
    lines = [' '.join(line.lower().strip().split()) for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem
    # in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    return article_lines, highlights


def write_to_tar(url_file, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the
       url_file and writes them to a out_file.
    """
    print("Making bin file for URLs listed in {}...".format(url_file))
    story_fnames = read_text_file(url_file)
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    with tarfile.open(out_file, 'w') as writer:
        for idx, s in enumerate(story_fnames):
            if idx % 1000 == 0:
                print("Writing story {} of {}; {:.2f} percent done".format(
                    idx, num_stories, float(idx)*100.0/float(num_stories)))

            # Look in the tokenized story dirs to find the .story file corresponding to this url
            if os.path.isfile(os.path.join(train_tokenized_stories_dir, s)):
                story_file = os.path.join(train_tokenized_stories_dir, s)
            elif os.path.isfile(os.path.join(valid_tokenized_stories_dir, s)):
                story_file = os.path.join(valid_tokenized_stories_dir, s)
            elif os.path.isfile(os.path.join(test_tokenized_stories_dir, s)):
                story_file = os.path.join(test_tokenized_stories_dir, s)
            else:
                print("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. \
                                          Was there an error during tokenization?" % (
                    s, train_tokenized_stories_dir, valid_tokenized_stories_dir))
                # Check again if tokenized stories directories contain correct number of files
                print("Checking that the tokenized stories directories %s and %s contain correct number of files..."
                      % (train_tokenized_stories_dir, valid_tokenized_stories_dir))
                raise Exception("Tokenized stories directories %s and %s contain correct number of files but story file \
                                          %s found in neither." % (
                train_tokenized_stories_dir, valid_tokenized_stories_dir, s))

            # Get the strings to write to .bin file
            article_sents, abstract_sents = get_art_abs(story_file)

            # Write to JSON file
            js_example = {}
            js_example['id'] = s.replace('.story', '')
            js_example['article'] = article_sents
            js_example['abstract'] = abstract_sents
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = ' '.join(article_sents).split()
                abs_tokens = ' '.join(abstract_sents).split()
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t != ""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file {}\n".format(out_file))

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"), 'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")


def split_text_into_sentences(stories_dir):
    print("%s: splitting sentences..." % stories_dir)
    count = 0
    for s in os.listdir(stories_dir):
        sentence = []
        with open(stories_dir+"/"+s, "r", encoding='utf-8') as file:
            text_data = file.readlines()
            for line in text_data:
                line = line.strip().replace(".", ". ").replace("?", "? ").replace("!", "! ")
                sentences = tokenizer.tokenize(line.strip())
                if len(sentences) > 0:
                    sentence += sentences
        with open(stories_dir+"/"+s, "w", encoding='utf-8') as file:
            file.write("\n\n".join(sentence))
        count += 1
        if count % 1000 == 0:
            print("writing %d files" % count)


if __name__ == '__main__':
    # Create some new directories
    if not os.path.exists(train_tokenized_stories_dir):
        os.makedirs(train_tokenized_stories_dir)
    if not os.path.exists(valid_tokenized_stories_dir):
        os.makedirs(valid_tokenized_stories_dir)
    if not os.path.exists(test_tokenized_stories_dir):
        os.makedirs(test_tokenized_stories_dir)
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    split_text_into_sentences(train_stories_dir)
    split_text_into_sentences(valid_stories_dir)
    split_text_into_sentences(test_stories_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_stories(train_stories_dir, train_tokenized_stories_dir)
    tokenize_stories(valid_stories_dir, valid_tokenized_stories_dir)
    tokenize_stories(test_stories_dir, test_tokenized_stories_dir)

    # Read the tokenized stories, do a little postprocessing
    # then write to bin files
    write_to_tar(all_test_urls, os.path.join(finished_files_dir, "test.tar"))
    write_to_tar(all_val_urls, os.path.join(finished_files_dir, "val.tar"))
    write_to_tar(all_train_urls, os.path.join(finished_files_dir, "train.tar"), makevocab=True)
