from srl.model_api import ModelAPI
from srl.preprocessors import Preprocessor
from utils.evaluators import Evaluator, calc_true_spans
from utils.loaders import Conll05Loader, CoNLL12Loader, load_emb
from utils.savers import save_pickle
from utils.misc import write, show_score_history, make_vocab_from_ids
from utils.misc import make_output_dir, get_file_names_in_dir, get_latest_param_fn


class Trainer(object):
    def __init__(self, argv):
        self.argv = argv
        self.model_api = ModelAPI(argv)
        self.preprocessor = Preprocessor(argv)
        self.evaluator = Evaluator(argv)
        self.loader = Conll05Loader(argv) if argv.data_type == "conll05" else CoNLL12Loader(argv)

        self.f1_history = {}
        self.best_valid_f1 = 0.0
        self.best_epoch = -1

    def train(self):
        write('\nTRAINING START\n')

        argv = self.argv
        loader = self.loader
        pproc = self.preprocessor

        make_output_dir(self.argv)

        #################
        # Load word emb #
        #################
        if argv.word_emb:
            write('Loading Word Embeddings...')
            word_list, word_emb = load_emb(argv.word_emb)
            vocab_word = pproc.make_vocab_word(word_list)
            write('\t- # Vocabs: %d' % vocab_word.size())
        else:
            vocab_word = word_emb = None

        #################
        # Load elmo emb #
        #################
        if self.argv.train_elmo_emb:
            write('Loading ELMo Embeddings...')
            train_elmo_emb = loader.load_hdf5(self.argv.train_elmo_emb)
        else:
            train_elmo_emb = None
        if self.argv.dev_elmo_emb:
            valid_elmo_emb = loader.load_hdf5(self.argv.dev_elmo_emb)
        else:
            valid_elmo_emb = None

        ###############
        # Load corpus #
        ###############
        write('Loading Corpus...')
        train_corpus = loader.load(path=argv.train_data,
                                   data_size=argv.data_size,
                                   is_test=False)
        valid_corpus = loader.load(path=argv.dev_data,
                                   data_size=argv.data_size,
                                   is_test=False)
        write('\t- # Sents: Train:%d  Valid:%d' % (len(train_corpus),
                                                   len(valid_corpus)))

        ##############
        # Make sents #
        ##############
        train_sents = pproc.make_sents(train_corpus)
        valid_sents = pproc.make_sents(valid_corpus)

        ###############
        # Make labels #
        ###############
        write('Making Labels...')

        if argv.save or argv.save_every_epoch:
            save_label = True
        else:
            save_label = False

        vocab_label_train = pproc.make_and_save_vocab_label(sents=train_sents,
                                                            vocab_label_init=None,
                                                            save=save_label,
                                                            load=True)
        vocab_label_valid = pproc.make_and_save_vocab_label(sents=valid_sents,
                                                            vocab_label_init=vocab_label_train,
                                                            save=False,
                                                            load=False)
        write('\t- # Labels: %d' % vocab_label_train.size())

        ###################
        # Set sent params #
        ###################
        train_sents = pproc.set_sent_params(sents=train_sents,
                                            elmo_emb=train_elmo_emb,
                                            vocab_word=vocab_word,
                                            vocab_label=vocab_label_train)
        valid_sents = pproc.set_sent_params(sents=valid_sents,
                                            elmo_emb=valid_elmo_emb,
                                            vocab_word=vocab_word,
                                            vocab_label=vocab_label_valid)

        ################
        # Make samples #
        ################
        write('Making Samples...')
        train_samples = pproc.make_samples(sents=train_sents,
                                           is_valid_data=False)
        valid_samples = pproc.make_samples(sents=valid_sents,
                                           is_valid_data=True)
        write('\t- # Samples: Train:%d  Valid:%d' % (len(train_samples),
                                                     len(valid_samples)))

        #################
        # Set Model API #
        #################
        if train_elmo_emb is not None:
            use_elmo = True
        else:
            use_elmo = False

        if argv.n_experts > 0:
            is_ensemble = True
        else:
            is_ensemble = False

        self.model_api.n_true_spans = calc_true_spans(train_sents)

        if is_ensemble:
            self.model_api.set_ensemble_model(word_emb=word_emb,
                                              use_elmo=use_elmo,
                                              vocab_word=vocab_word,
                                              vocab_label=vocab_label_train,
                                              vocab_label_valid=vocab_label_valid)
            self.model_api.load_experts_params(argv.load_param_dir)
            self.model_api.set_init_ensemble_param()
            self.model_api.set_ensemble_train_func()
            if self.model_api.vocab_label_valid:
                self.model_api.set_ensemble_pred_func()
            init_epoch = 0
        else:
            self.model_api.set_model(word_emb=word_emb,
                                     use_elmo=use_elmo,
                                     vocab_word=vocab_word,
                                     vocab_label=vocab_label_train,
                                     vocab_label_valid=vocab_label_valid)
            if argv.load_param_latest:
                if argv.output_dir:
                    dir_name = argv.output_dir
                else:
                    dir_name = 'output'
                param_fns = get_file_names_in_dir(dir_path=dir_name,
                                                  prefix='param')
                opt_param_fns = get_file_names_in_dir(dir_path=dir_name,
                                                      prefix='opt')
                param_fn, latest_epoch = get_latest_param_fn(file_names=param_fns)
                opt_param_fn, _ = get_latest_param_fn(file_names=opt_param_fns)
                self.model_api.argv.load_param = param_fn
                self.model_api.argv.load_opt_param = opt_param_fn
                self.model_api.load_params(param_fn)
                init_epoch = latest_epoch + 1
            elif argv.load_param:
                self.model_api.load_params(argv.load_param)
                init_epoch = 0
            else:
                init_epoch = 0

            self.model_api.set_train_func()
            if self.model_api.vocab_label_valid:
                self.model_api.set_pred_func()

        #######################
        # Run training epochs #
        #######################
        self._run_epochs(train_samples, valid_samples, init_epoch)

    def _run_epochs(self, train_samples, valid_samples=None, init_epoch=0):
        write('\nTRAIN START')

        argv = self.argv
        pproc = self.preprocessor
        vocab_label_valid = self.model_api.vocab_label_valid

        if valid_samples:
            valid_batches = pproc.make_batches(samples=valid_samples,
                                               is_valid_data=True)
            valid_batch_x, valid_batch_y = pproc.split_x_and_y(valid_batches)

        ##########################################
        # Initial result with pre-trained params #
        ##########################################
        if (argv.load_param or argv.load_param_dir) and valid_samples:
            write('\nEpoch: 0 (Using the Pre-trained Params)')
            write('VALID')
            valid_batch_y_pred = self.model_api.predict(valid_batch_x)
            self.best_valid_f1 = self.evaluator.f_score(y_true=valid_batch_y,
                                                        y_pred=valid_batch_y_pred,
                                                        vocab_label=vocab_label_valid)

        #############
        # Main loop #
        #############
        for epoch in range(init_epoch, argv.epoch):
            write('\nEpoch: %d' % (epoch + 1))
            write('TRAIN')

            if argv.halve_lr and epoch > 49 and (epoch % 25) == 0:
                lr = self.model_api.optimizer.lr.get_value(borrow=True)
                self.model_api.optimizer.lr.set_value(lr * 0.5)
                write('### HALVE LEARNING RATE: %f -> %f' % (lr, lr * 0.5))

            ############
            # Training #
            ############
            train_batches = pproc.make_batches(train_samples)
            self.model_api.train(train_batches)

            ##############
            # Validating #
            ##############
            if valid_samples:
                write('VALID')
                valid_batch_y_pred = self.model_api.predict(valid_batch_x)
                valid_f1 = self.evaluator.f_score(y_true=valid_batch_y,
                                                  y_pred=valid_batch_y_pred,
                                                  vocab_label=vocab_label_valid)
                if self.best_valid_f1 < valid_f1:
                    self.best_valid_f1 = valid_f1
                    self.best_epoch = epoch
                    self.f1_history[self.best_epoch + 1] = [self.best_valid_f1]

                    if argv.save:
                        self.model_api.save_params(epoch=0)
                        self.model_api.optimizer.save_params(epoch=0)

            if argv.save_every_epoch:
                self.model_api.save_params(epoch=epoch)
                self.model_api.optimizer.save_params(epoch=epoch)

            show_score_history(self.f1_history)

    def validate(self):
        write('\nVALIDATING START\n')

        argv = self.argv
        loader = self.loader
        pproc = self.preprocessor

        make_output_dir(self.argv)

        ################
        # Load dataset #
        ################
        write('Loading Dataset...')
        valid_corpus = loader.load(path=argv.dev_data,
                                   data_size=argv.data_size,
                                   is_test=False)
        valid_sents = pproc.make_sents(valid_corpus)

        #################
        # Load init emb #
        #################
        if argv.word_emb:
            write('Loading Embeddings...')
            word_list_emb, word_emb = load_emb(argv.init_emb)
            vocab_word = pproc.make_vocab_word(word_list_emb)
            write('\t- # Embedding Words: %d' % vocab_word.size())
        else:
            vocab_word = word_emb = None

        #################
        # Load elmo emb #
        #################
        if argv.dev_elmo_emb:
            write('Loading ELMo Embeddings...')
            valid_elmo_emb = loader.load_hdf5(argv.dev_elmo_emb)
        else:
            valid_elmo_emb = None

        ###############
        # Make labels #
        ###############
        write('Loading Labels...')
        label_key_value = loader.load_key_value_format(argv.load_label)
        vocab_label_train = make_vocab_from_ids(label_key_value)
        vocab_label_valid = pproc.make_and_save_vocab_label(sents=valid_sents,
                                                            vocab_label_init=vocab_label_train,
                                                            save=False,
                                                            load=False)
        write('\t- # Labels: %d' % vocab_label_train.size())

        ###################
        # Set sent params #
        ###################
        valid_sents = pproc.set_sent_params(sents=valid_sents,
                                            elmo_emb=valid_elmo_emb,
                                            vocab_word=vocab_word,
                                            vocab_label=vocab_label_valid)
        ################
        # Make samples #
        ################
        write('Making Valid Samples...')
        valid_samples = pproc.make_samples(sents=valid_sents)
        valid_batches = pproc.make_batches(samples=valid_samples,
                                           is_valid_data=True)
        valid_batch_x, y_true = pproc.split_x_and_y(valid_batches)
        write('\t- # Samples: %d' % len(valid_samples))

        #################
        # Set Model API #
        #################
        if valid_elmo_emb is not None:
            use_elmo = True
        else:
            use_elmo = False

        self.model_api.set_model(word_emb=word_emb,
                                 use_elmo=use_elmo,
                                 vocab_word=vocab_word,
                                 vocab_label=vocab_label_train,
                                 vocab_label_valid=vocab_label_valid)
        self.model_api.set_pred_func()

        ############
        # Validate #
        ############
        write('\nVALIDATION START')

        dir_name = argv.load_param_dir
        param_files = [fn for fn in get_file_names_in_dir(dir_name)
                       if fn.split('/')[-1].startswith('param')]
        write('\t- # Param Files: %d' % len(param_files))

        best_file = None
        best_param = None
        best_f1 = -1.0

        for param_file in param_files:
            write('\nFile Name: %s' % param_file)
            self.model_api.load_params(param_file)
            y_pred = self.model_api.predict(valid_batch_x)
            valid_f1 = self.evaluator.f_score(y_true=y_true,
                                              y_pred=y_pred,
                                              vocab_label=vocab_label_valid)
            if best_f1 < valid_f1:
                best_f1 = valid_f1
                best_file = param_file
                best_param = [p.get_value(borrow=True)
                              for p in self.model_api.model.params]

        write('Best Param=%s F1=%f' % (best_file, best_f1))

        fn = argv.load_param_dir
        if argv.output_fn:
            fn += '/param.%s' % argv.output_fn
        else:
            fn += '/param.%s.best' % argv.method
        save_pickle(fn=fn, data=best_param)
