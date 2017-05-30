'''
Created on Mar 20, 2016
'''


from theano import tensor
from blocks.bricks.lookup import LookupTable
from blocks.bricks.base import application
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.search import CandidateNotFoundError
from lvsr.dependency.recognizer import Bottom
from lvsr.skladnica.glove import CharacetrToWordEmbeddings, FeaturesToWordEmbeddings
from blocks.bricks.wrappers import WithExtraDims

class NDCharacetrToWordEmbeddings(CharacetrToWordEmbeddings):
    decorators = [WithExtraDims()]

class NDFeaturesToWordEmbeddings(FeaturesToWordEmbeddings):
    decorators = [WithExtraDims()]
    
def count_errors(first, second):
    min_len = min(len(first), len(second))
    max_len = max(len(first), len(second))
    err_count = max_len - min_len
    for i in xrange(min_len):
        if first[i] != second[i]:
            err_count += 1
    return err_count / float(max_len)


class DependencyBottom(Bottom):
    def __init__(self, input_sources_dims,
                 dims=None, activation=None,
                 char_to_word_conf={}, feats_to_word_conf={}, lang_postfix="",
                 **kwargs):
        additional_sources = kwargs.pop('additional_sources')
        self.pointers_soften = kwargs.pop('pointers_soften', False)
        embedding_dims = {}
        for ed, val in kwargs.iteritems():
            if 'embedding_dim' in ed:
                embedding_dims[ed] = val
        for ed in embedding_dims.keys():
            kwargs.pop(ed)
        super(DependencyBottom, self).__init__(**kwargs)
        assert not dims
        assert not activation
        self.input_sources_dims = input_sources_dims
        self.input_sources = sorted(self.input_sources)
        self.components = {}
        self.output_dim = 0
        self.batch_inputs = {}
        self.single_inputs = {}
        self.additional_sources_inputs = {source: tensor.imatrix(source)
                                          for source in additional_sources}
        self.single_additional_sources_inputs = {source: tensor.ivector(source)
                                                 for source in additional_sources}
        if self.pointers_soften:
            self.additional_sources_inputs['pointers'+lang_postfix] = tensor.ftensor3('pointers'+lang_postfix)
            self.single_additional_sources_inputs['pointers'+lang_postfix] = tensor.fmatrix('pointers'+lang_postfix)
        self.mask_input = tensor.matrix(self.input_sources[0] + "_mask")
        for source in self.input_sources:
            if lang_postfix:
                source_name = source[:-len(lang_postfix)]
            else:
                source_name = source
            source_dim = input_sources_dims[source]
            if '_per_' in source_name and 'features' not in source_name:
                self.components[source] = NDCharacetrToWordEmbeddings(
                        source_dim, name=source_name[:4]+"_embedder", **char_to_word_conf)
                self.components[source]._extra_ndim = 1
                self.output_dim += self.components[source].output_dim
                self.batch_inputs[source] = tensor.itensor3(source)
                self.single_inputs[source] = tensor.imatrix(source)
            elif 'features_per_word' in source_name:
                self.components[source] = NDFeaturesToWordEmbeddings(
                        source_dim, name=source_name[:4]+"_embedder", **feats_to_word_conf)
                self.components[source]._extra_ndim = 1
                self.output_dim += self.components[source].output_dim
                self.batch_inputs[source] = tensor.itensor3(source)
                self.single_inputs[source] = tensor.imatrix(source)
            elif source_name+'_embedding_dim' in embedding_dims.keys():
                embed_dim = embedding_dims[source + '_embedding_dim']
                assert embed_dim
                self.components[source] = LookupTable(source_dim,
                                                      embed_dim,
                                                      name=source + "_lut")
                self.output_dim += embed_dim
                self.batch_inputs[source] = tensor.imatrix(source)
                self.single_inputs[source] = tensor.ivector(source)
            else:
                raise Exception("Unknown source type: {}".format(source))
        assert self.components
        self.children.extend(self.components.values())
        #from IPython import embed; embed()

    @application
    def apply(self, **sources):
        ret_components = []
        for source in self.input_sources:
            s = sources.pop(source)
            ret_components.append(
                self.components[source].apply(s))
        assert not sources
        return tensor.concatenate(ret_components, axis=2)

    @application
    def batch_size(self, **sources):
        # import IPython; IPython.embed()
        return sources[self.input_sources[0]].shape[1]

    @application
    def num_time_steps(self, **sources):
        # import IPython; IPython.embed()
        return sources[self.input_sources[0]].shape[0]

    def get_batch_additional_sources(self):
        return self.additional_sources_inputs

    def get_single_additional_sources(self):
        return self.single_additional_sources_inputs

    def get_batch_inputs(self):
        return self.batch_inputs

    def get_mask(self):
        return self.mask_input

    def get_single_sequence_inputs(self):
        return self.single_inputs

    def single_to_batch_inputs(self, inputs):
        # Note: this code supports many inputs, which are all sequences
        inputs = {n: v[:, None, :] if v.ndim == 2 else v[:, None]
                  for (n, v) in inputs.items()}
        inputs_mask = tensor.ones((self.num_time_steps(**inputs),
                                   self.batch_size(**inputs)))
        return inputs, inputs_mask


class AuxiliaryErrorRates(MonitoredQuantity):
    """
    Monitored Quantity is limited to returning only one quantity.

    This class looks into attributes of a DependencyErrorRate instance to
    retrieve the required quantities.
    """
    def __init__(self, dep, **kwargs):
        super(AuxiliaryErrorRates, self).__init__(**kwargs)
        self.dep = dep

    def initialize(self):
        pass

    def accumulate(self, *args):
        pass

    def readout(self):
        return getattr(self.dep, self.name)


class DependencyErrorRate(MonitoredQuantity):
    def __init__(self, recognizer, data,
                 beam_size,  char_discount, round_to_inf, stop_on,
                 **kwargs):
        self.recognizer = recognizer
        self.beam_size = beam_size
        self.char_discount = char_discount
        self.round_to_inf = round_to_inf
        self.stop_on = stop_on
        # Will only be used to decode generated outputs,
        # which is necessary for correct scoring.
        self.data = data
        kwargs.setdefault('name', 'UAS')
        requires_default = self.recognizer.single_inputs.values()
        self.inputs_count = len(requires_default)
        requires_default += [self.recognizer.single_additional_sources[self.recognizer.pointers_name]]
        #if hasattr(self.recognizer, 'single_additional_sources'):
        #    requires_default += self.recognizer.single_additional_sources.values()
        requires_default += [self.recognizer.single_labels]
        kwargs.setdefault('requires', requires_default)
        super(DependencyErrorRate, self).__init__(**kwargs)

        self.recognizer.init_beam_search(self.beam_size)

    def initialize(self):
        self.total_UAS_errs = 0.
        self.total_LAB_errs = 0.
        self.total_LAS_errs = 0.
        self.UAS_errs = 0.
        self.LAB_errs = 0.
        self.LAS_errs = 0.
        self.total_errors = 0.
        self.total_length = 0.
        self.num_examples = 0

    def accumulate(self, *args):
        
        input_vars = self.requires[:self.inputs_count]
        beam_inputs = {var.name: val for var, val in zip(input_vars,
                                                         args[:-1])}
        transcription = args[-1]
        pointers = args[-2]
        if pointers.ndim == 2:
            pointers = pointers.argmax(axis=1)
        # Hack to avoid hopeless decoding of an untrained model
        if self.num_examples > 10 and self.UAS > 0.8:
            self.UAS_errs = 1.
            self.LAB_errs = 1.
            self.LAS_errs = 1.
            return
        data = self.data
        try:
            outputs, _, _ = self.recognizer.beam_search(
                beam_inputs,
                char_discount=self.char_discount,
                round_to_inf=self.round_to_inf,
                stop_on=self.stop_on,
                validate_solution_function=getattr(data.info_dataset,
                                                   'validate_solution',
                                                   None)
                )
            recognized = outputs[0]
            recognized_pointers = outputs[1]
            ptr_errs = (pointers[1:-1] != recognized_pointers[1:-1])
            label_errs = (transcription[1:-1] != recognized[1:-1])

            UAS_errs = ptr_errs.sum()
            LAB_errs = label_errs.sum()
            LAS_errs = (ptr_errs | label_errs).sum()

        except CandidateNotFoundError:
            UAS_errs = LAB_errs = LAS_errs = pointers.shape[0]-2

        self.total_UAS_errs += UAS_errs
        self.total_LAB_errs += LAB_errs
        self.total_LAS_errs += LAS_errs
        self.total_length += pointers.shape[0]-2
        self.num_examples += 1
        self.LAS = self.total_LAS_errs/self.total_length
        self.UAS = self.total_UAS_errs/self.total_length
        self.LAB = self.total_LAB_errs/self.total_length

    def readout(self):
        return self.UAS
