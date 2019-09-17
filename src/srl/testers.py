from utils.loaders import load_emb
from utils.misc import write, make_vocab_from_ids


class Tester(object):
    def __init__(self,
                 argv,
                 loader,
                 saver,
                 preprocessor,
                 evaluator,
                 model_api):
        self.argv = argv
        self.loader = loader
        self.saver = saver
        self.preprocessor = preprocessor
        self.evaluator = evaluator
        self.model_api = model_api

    def predict(self):
        argv = self.argv
        pproc = self.preprocessor
        loader = self.loader

        ################
        # Load dataset #
        ################
        write('Loading Dataset...')
        test_corpus = loader.load(path=argv.test_data,
                                  data_size=argv.data_size,
                                  is_test=True)
        test_sents = pproc.make_sents(test_corpus)

        #################
        # Load init emb #
        #################
        if argv.word_emb:
            write('Loading Embeddings...')
            word_list, word_emb = load_emb(argv.word_emb)
            vocab_word = pproc.make_vocab_word(word_list)
            write('\t- # Embedding Words: %d' % vocab_word.size())
        else:
            vocab_word = word_emb = None

        if argv.test_elmo_emb:
            write('Loading ELMo Embeddings...')
            test_elmo_emb = loader.load_hdf5(argv.test_elmo_emb)
        else:
            test_elmo_emb = None

        ###############
        # Make labels #
        ###############
        label_key_value = loader.load_key_value_format(argv.load_label)
        vocab_label = make_vocab_from_ids(label_key_value)
        write('\t- # Labels: %d' % vocab_label.size())

        ###################
        # Set sent params #
        ###################
        test_sents = pproc.set_sent_config(sents=test_sents,
                                           elmo_emb=test_elmo_emb,
                                           vocab_word=vocab_word,
                                           vocab_label=None)
        ################
        # Make samples #
        ################
        write('Making Test Samples...')
        test_batches = pproc.make_batch_per_sent(sents=test_sents)
        write('\t- # Test Samples: %d' % len(test_batches))

        #############
        # Model API #
        #############
        use_elmo = True if test_elmo_emb is not None else False

        if argv.n_experts > 0:
            self.model_api.set_ensemble_model(word_emb=word_emb,
                                              use_elmo=use_elmo,
                                              vocab_word=vocab_word,
                                              vocab_label=vocab_label,
                                              vocab_label_valid=None)
            self.model_api.load_params(argv.load_param)
            self.model_api.load_experts_params(argv.load_param_dir)
            self.model_api.set_ensemble_pred_func()
        else:
            self.model_api.set_model(word_emb=word_emb,
                                     use_elmo=use_elmo,
                                     vocab_word=vocab_word,
                                     vocab_label=vocab_label,
                                     vocab_label_valid=None)
            self.model_api.load_params(argv.load_param)
            self.model_api.set_pred_func()

        ###########
        # Testing #
        ###########
        write('\nPREDICTION START')
        test_y_pred = self.model_api.predict(test_batches)
        self.saver.save_props(corpus=test_sents,
                              labels=test_y_pred,
                              vocab_label=vocab_label)
        self.saver.save_json_format(corpus=test_sents,
                                    labels=test_y_pred,
                                    vocab_label=vocab_label)
