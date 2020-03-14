"""SiamRPN network
Code adapted from https://github.com/STVIR/pysot"""
# coding:utf-8
# pylint: disable=arguments-differ,unused-argument
from mxnet.gluon.block import HybridBlock
import mxnet as mx
from gluoncv.model_zoo.siamrpn.siam_alexnet import alexnetlegacy
from gluoncv.model_zoo.siamrpn.siam_rpn import DepthwiseRPN
from mxnet.context import cpu
import pdb


class SiamRPN(HybridBlock):
    """SiamRPN"""
    def __init__(self, bz=1, if_train=False, ctx=cpu(), **kwargs):
        super(SiamRPN, self).__init__(**kwargs)
        self.backbone = alexnetlegacy(ctx=ctx)
        self.rpn_head = DepthwiseRPN(bz=bz, if_train=if_train, ctx=ctx)
        self.bz = bz
        self.if_train = if_train
        self.zbranch = None
        self.xbranch = None
        self.cls = None
        self.loc = None


    def template(self, zinput):
        """template z branch"""
        zbranch = self.backbone(zinput)
        self.zbranch = zbranch

    def track(self, xinput):
        """track x branch

        Parameters
        ----------
            xinput : np.ndarray
                predicted frame

        Returns
        -------
        dic
            predicted frame result
        """
        xbranch = self.backbone(xinput)
        self.cls, self.loc = self.rpn_head(self.zbranch, xbranch)
        return {
            'cls': self.cls,
            'loc': self.loc,
            }

    def hybrid_forward(self, F, zinput, xinput):
        """ Hybrid forward of SiamRPN net
            only used in training """
        zbranch = self.backbone(zinput)#(32,256,6,6)
        xbranch = self.backbone(xinput)#(32,256,22,22)
        self.cls, self.loc = self.rpn_head(zbranch, xbranch)
        #return self.cls, self.loc
        return tuple([self.cls, self.loc])


def get_Siam_RPN(base_name, bz=1, if_train=False, pretrained=False, ctx=mx.cpu(0),
                 root='~/.mxnet/models', **kwargs):
    """get Siam_RPN net and get pretrained model if have pretrained

    Parameters
    ----------
    base_name : str
        Backbone model name
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    HybridBlock
        A SiamRPN Tracking network.
    """
    net = SiamRPN(bz=bz, if_train=if_train, ctx=ctx)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file('siamrpn_%s'%(base_name),
                                           tag=pretrained, root=root), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    return net

def siamrpn_alexnet_v2_otb15(**kwargs):
    """Alexnet backbone model from
    `"High Performance Visual Tracking with Siamese Region Proposal Network
        Object tracking"
    <http://openaccess.thecvf.com/content_cvpr_2018/papers/
    Li_High_Performance_Visual_CVPR_2018_paper.pdf>`_ paper.
    """
    return get_Siam_RPN('alexnet_v2_otb15', **kwargs)
