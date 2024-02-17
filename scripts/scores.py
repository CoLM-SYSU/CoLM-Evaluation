# -*- coding: utf-8 -*-
import numpy as np
import sys
import xarray as xr
import numpy as np

class scores_geo:
    def __init__(self):
        
        """
        A libray with Python functions for calculations of
        micrometeorological parameters and some miscellaneous
        utilities.
        functions:
        apb                 : absolute percent bias
        rmse                : root mean square error
        mae                 : mean absolute error
        bias                : bias
        pc_bias             : percentage bias
        NSE                 : Nash-Sutcliffe Coefficient
        L                   : likelihood estimation
        correlation         : correlation
        correlation_R2      : correlation**2, R2
        index_agreement     : index of agreement
        KGE                 : Kling-Gupta Efficiency
        
        """
        self.name = 'scores'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        np.seterr(all='ignore')  
        # Turn off only the RuntimeWarning
        #np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')

    def KGESS(self,s,o):
        """
        Normalized Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kgess:Normalized Kling-Gupta Efficiency
        note:
        KGEbench= −0.41 from Knoben et al., 2019)
        Knoben, W. J. M., Freer, J. E., and Woods, R. A.: Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–
        Gupta efficiency scores, Hydrol. Earth Syst. Sci., 23, 4323–4331,
        https://doi.org/10.5194/hess-23-4323-2019, 2019.
        """
        kge=self.KGE(s,o)
        kgess=(kge-(-0.41))/(1.0-(-0.41))
        return kgess   #, cc, alpha, beta


    def nBiasScore(self,s,o):
        """
        Bias Score from ILAMB
        input:
            s: simulated
            o: observed
        output:
            nbiasscore: normalized Bias Score
            ranging from 0 to 1, with 1 being the best
        """
        #bias = np.mean(s)-np.mean(o)
        #crms = np.sqrt(np.mean((s-np.mean(o))**2))
        #biasscore = np.abs(bias)/crms
        #nbiasscore=np.exp(-biasscore)
        bias       = s.mean(dim='time')-o.mean(dim='time')
        crms       = np.sqrt(((s-o.mean(dim='time'))**2).mean(dim='time'))
        biasscore  = np.abs(bias)/crms
        nbiasscore = np.exp(-biasscore)
        return nbiasscore
    
    def nRMSEScore(self,s,o):
        """
        RMSE Score from ILAMB
        input:
            s: simulated
            o: observed
        output:
            rmse: normalized RMSE Score
            ranging from 0 to 1, with 1 being the best
        """

        #crmse = np.sqrt(np.mean(((s-np.mean(s))-(o-np.mean(o)))**2))
        #crms  = np.sqrt(np.mean((s-np.mean(o))**2))
        #RMSEScore = crmse/crms
        #nRMSEScore=np.exp(-RMSEScore)
        crmse = np.sqrt((((s-s.mean(dim='time'))-(o-o.mean(dim='time')))**2).mean(dim='time'))
        crms       = np.sqrt(((s-o.mean(dim='time'))**2).mean(dim='time'))
        RMSEScore = crmse/crms
        nRMSEScore=np.exp(-RMSEScore)
        return nRMSEScore

    ##not ready yet
    ##### for forcast verification

    def BSS(self,s,o,threshold=0.1):
        '''
        BSS - Brier skill score
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: BSS value
        '''
        o = np.where(o >= threshold, 1, 0)
        s = np.where(s >= threshold, 1, 0)

        o = o.flatten()
        s = s.flatten()

        return np.sqrt(np.mean((o - s) ** 2))

    def HSS(self,s,o,threshold=0.1):
        '''
        HSS - Heidke skill score
        Args:
            o (numpy.ndarray): observations
            pre (numpy.ndarray): pre
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: HSS value
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(s,o,threshold=threshold)

        HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
        HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
                   (misses + falsealarms)*(hits + correctnegatives))

        return HSS_num / HSS_den

    def BIAS(self,s,o,  threshold = 0.1):
        '''
        func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses) 
              alias: (TP + FP)/(TP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float

        偏差评分(Bias score)主要用来衡量模式对某一量级降水的预报偏差, 
        该评分在数值上等于预报区域内满足某降水阈值的总格点数与对应实况降水总格点数的比值(Kong et al, 2008)。
        用来反映降水总体预报效果的检验方法。
        Bias = y_pred_1/y_obs_1 = (hits + falsealarms)/(hits + misses)
        '''    
        hits, misses, falsealarms, correctnegatives = self.prep_clf(s,o,threshold=threshold)

        return (hits + falsealarms) / (hits + misses)

    def POD(self,s,o,threshold=0.1):
        '''
        func : 计算命中率 hits / (hits + misses)
        pod - Probability of Detection
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: PDO value

        Probability of Detection。即预测出的实际的降水区域占据全部实际降水区域的比重。
        POD =  hits / y_obs_1  = hits / (hits + misses) = 1- MAR
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(s,o,threshold=threshold)
        return hits / (hits + misses)

    def MAR(self,s,o, threshold=0.1):
        '''
        func : 计算漏报率 misses / (hits + misses)
        MAR - Missing Alarm Rate
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: MAR value

        Missing Alarm Rate。实际降水区域中漏报的区域占据全部实际降水区域的比重。
        MAR =  (y_obs_1 - hits)/y_obs_1 =  misses / (hits + misses)
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(s,o,threshold=threshold)

        return misses / (hits + misses)

    def FAR(self,s,o, threshold=0.1):
        '''
        func: 计算误警率。falsealarms / (hits + falsealarms) 
        FAR - false alarm rate
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
        float: FAR value
    
        False Alarm Rate 。在预报降水区域中实际没有降水的区域占总预报降水区域的比重。
        FAR =  (y_pre_1 - hits)/y_pre_1 =  falsealarms / (hits + falsealarms)
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf( s,o,threshold=threshold)

        return falsealarms / (hits + falsealarms)

    def ETS(self,s,o, threshold=0.1):
        '''
        ETS - Equitable Threat Score
        details in the paper:
        Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
        radar-derived precipitation with model-derived winds.
        Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
        Returns:
            float: ETS value
        公平技巧评分(Equitable Threat Score, ETS)用于衡量对流尺度集合预报的预报效果。
        ETS评分表示在预报区域内满足某降水阈值的降水预报结果相对于满足同样降水阈值的随机预报的预报技巧；
        ETS评分是对TS评分的改进，能对空报或漏报进行惩罚，使评分相对后者更加公平.
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(s,o,threshold=threshold)
        num = (hits + falsealarms) * (hits + misses)
        den = hits + misses + falsealarms + correctnegatives
        Dr = num / den

        ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

        return ETS

    def TS(self,s,o, threshold=0.1):
        '''
        func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
              alias: TP/(TP+FP+FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float

        TS：风险评分ThreatScore;

        CSI:  critical success index 临界成功指数;
        两者的物理概念完全一致。
        '''

        hits, misses, falsealarms, correctnegatives = self.prep_clf( s,o,threshold=threshold)

        return hits/(hits + falsealarms + misses) 

    def prep_clf(self,s,o,threshold=0.1):
        '''
        func: 计算二分类结果-混淆矩阵的四个元素
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

        returns:
            hits, misses, falsealarms, correctnegatives
            #aliases: TP, FN, FP, TN 
        '''
        #根据阈值分类为 0, 1
        o = np.where(o >= threshold, 1, 0)
        s = np.where(s >= threshold, 1, 0)

        # True positive (TP)
        hits = np.sum((o == 1) & (s == 1))

        # False negative (FN)
        misses = np.sum((o == 1) & (s == 0))

        # False positive (FP)
        falsealarms = np.sum((o == 0) & (s == 1))

        # True negative (TN)
        correctnegatives = np.sum((o == 0) & (s == 0))

        return hits, misses, falsealarms, correctnegatives

    def precision(self,s,o,threshold=0.1):
        '''
        func: 计算精确度precision: TP / (TP + FP)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

        returns:
            dtype: float
        '''

        TP, FN, FP, TN = self.prep_clf(s,o,threshold=threshold)

        return TP / (TP + FP)

    def recall(self,s,o, threshold=0.1):
        '''
        func: 计算召回率recall: TP / (TP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float
        '''

        TP, FN, FP, TN = self.prep_clf(s,o,threshold=threshold)

        return TP / (TP + FN)

    def ACC(self,s,o, threshold=0.1):
        '''
        func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float
        '''

        TP, FN, FP, TN = self.prep_clf(s,o,threshold=threshold)

        return (TP + TN) / (TP + TN + FP + FN)

    def FSC(self,s,o, threshold=0.1):
        '''
        func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
        '''
        precision_socre = self.precision(s,o,threshold=threshold)
        recall_score = self.recall(s,o,threshold=threshold)

        return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))

class scores_stn:
    def __init__(self,s= np.array([]),o= np.array([])):
        
        """
        A libray with Python functions for calculations of
        micrometeorological parameters and some miscellaneous
        utilities.
        functions:
        apb                 : absolute percent bias
        rmse                : root mean square error
        mae                 : mean absolute error
        bias                : bias
        pc_bias             : percentage bias
        NSE                 : Nash-Sutcliffe Coefficient
        L                   : likelihood estimation
        correlation         : correlation
        correlation_R2      : correlation**2, R2
        index_agreement     : index of agreement
        KGE                 : Kling-Gupta Efficiency
        
        """
        self.name = 'scores'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        s = s.flatten()
        o = o.flatten()
        #check if the length of the vectors are same or not
        if len(s) != len(o):
            print("Length of both the vectors must be same")
            sys.exit(1)
        else:
            #removed the data from simulated and observed data
            #   whereever the observed data contains nan
            data   = np.array([s,o])
            data   = np.transpose(data)
            data   = data[~np.isnan(data).any(1)]
            self.s = data[:,0]
            self.o = data[:,1]
        
        np.seterr(all='ignore')  #
        # Turn off only the RuntimeWarning
        #np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')


    def KGESS(self):
        """
        Normalized Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kgess:Normalized Kling-Gupta Efficiency
        note:
        KGEbench= −0.41 from Knoben et al., 2019)
        Knoben, W. J. M., Freer, J. E., and Woods, R. A.: Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–
        Gupta efficiency scores, Hydrol. Earth Syst. Sci., 23, 4323–4331,
        https://doi.org/10.5194/hess-23-4323-2019, 2019.
        """
        cc  = self.correlation()
        alpha = np.std(self.s)/np.std(self.o)
        beta = np.sum(self.s)/np.sum(self.o)
        kge = 1- np.sqrt( (cc-1)**2 + (alpha-1)**2 + (beta-1)**2 )
        kgess=(kge-(-0.41))/(1.0-(-0.41))
        return kgess   #, cc, alpha, beta

    def nBiasScore(self):
        """
        Bias Score from ILAMB
        input:
            s: simulated
            o: observed
        output:
            nbiasscore: normalized Bias Score
            ranging from 0 to 1, with 1 being the best
        """
        bias = np.mean(self.s)-np.mean(self.o)
        crms = np.sqrt(np.mean((self.s-np.mean(self.o))**2))
        biasscore = np.abs(bias)/crms
        nbiasscore=np.exp(-biasscore)
        return nbiasscore
    
    def nRMSEScore(self):
        """
        RMSE Score from ILAMB
        input:
            s: simulated
            o: observed
        output:
            rmse: normalized RMSE Score
            ranging from 0 to 1, with 1 being the best
        """
        crmse = np.sqrt(np.mean(((self.s-np.mean(self.s))-(self.o-np.mean(self.o)))**2))
        crms  = np.sqrt(np.mean((self.s-np.mean(self.o))**2))
        RMSEScore = crmse/crms
        nRMSEScore=np.exp(-RMSEScore)
        return nRMSEScore

##### for forcast verification

    def BSS(self,threshold=0.1):
        '''
        BSS - Brier skill score
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: BSS value
        '''
        self.o = np.where(self.o >= threshold, 1, 0)
        self.s = np.where(self.s >= threshold, 1, 0)

        self.o = self.o.flatten()
        self.s = self.s.flatten()

        return np.sqrt(np.mean((self.o - self.s) ** 2))

    def HSS(self,threshold=0.1):
        '''
        HSS - Heidke skill score
        Args:
            self.o (numpy.ndarray): observations
            pre (numpy.ndarray): pre
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: HSS value
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(threshold=threshold)

        HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
        HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
                   (misses + falsealarms)*(hits + correctnegatives))

        return HSS_num / HSS_den

    def BIAS(self,  threshold = 0.1):
        '''
        func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses) 
              alias: (TP + FP)/(TP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float

        偏差评分(Bias score)主要用来衡量模式对某一量级降水的预报偏差, 
        该评分在数值上等于预报区域内满足某降水阈值的总格点数与对应实况降水总格点数的比值(Kong et al, 2008)。
        用来反映降水总体预报效果的检验方法。
        Bias = y_pred_1/y_obs_1 = (hits + falsealarms)/(hits + misses)
        '''    
        hits, misses, falsealarms, correctnegatives = self.prep_clf(threshold=threshold)

        return (hits + falsealarms) / (hits + misses)

    def POD(self,threshold=0.1):
        '''
        func : 计算命中率 hits / (hits + misses)
        pod - Probability of Detection
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: PDO value

        Probability of Detection。即预测出的实际的降水区域占据全部实际降水区域的比重。
        POD =  hits / y_obs_1  = hits / (hits + misses) = 1- MAR
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(threshold=threshold)
        return hits / (hits + misses)

    def MAR(self, threshold=0.1):
        '''
        func : 计算漏报率 misses / (hits + misses)
        MAR - Missing Alarm Rate
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
            float: MAR value

        Missing Alarm Rate。实际降水区域中漏报的区域占据全部实际降水区域的比重。
        MAR =  (y_obs_1 - hits)/y_obs_1 =  misses / (hits + misses)
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(threshold=threshold)

        return misses / (hits + misses)

    def FAR(self, threshold=0.1):
        '''
        func: 计算误警率。falsealarms / (hits + falsealarms) 
        FAR - false alarm rate
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                 (rain/no rain)
        Returns:
        float: FAR value
    
        False Alarm Rate 。在预报降水区域中实际没有降水的区域占总预报降水区域的比重。
        FAR =  (y_pre_1 - hits)/y_pre_1 =  falsealarms / (hits + falsealarms)
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf( threshold=threshold)

        return falsealarms / (hits + falsealarms)

    def ETS(self, threshold=0.1):
        '''
        ETS - Equitable Threat Score
        details in the paper:
        Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
        radar-derived precipitation with model-derived winds.
        Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
        Returns:
            float: ETS value
        公平技巧评分(Equitable Threat Score, ETS)用于衡量对流尺度集合预报的预报效果。
        ETS评分表示在预报区域内满足某降水阈值的降水预报结果相对于满足同样降水阈值的随机预报的预报技巧；
        ETS评分是对TS评分的改进，能对空报或漏报进行惩罚，使评分相对后者更加公平.
        '''
        hits, misses, falsealarms, correctnegatives = self.prep_clf(threshold=threshold)
        num = (hits + falsealarms) * (hits + misses)
        den = hits + misses + falsealarms + correctnegatives
        Dr = num / den

        ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

        return ETS

    def TS(self, threshold=0.1):

        '''
        func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
              alias: TP/(TP+FP+FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float

        TS：风险评分ThreatScore;

        CSI:  critical success index 临界成功指数;

        两者的物理概念完全一致。

        如下图：y_pre_1为预测的降水区( >= threshold，下同)，y_obs_1为观测的降水区，hits为两者交界区，
        '''

        hits, misses, falsealarms, correctnegatives = self.prep_clf( threshold=threshold)

        return hits/(hits + falsealarms + misses) 

    def prep_clf(self,threshold=0.1):
        '''
        func: 计算二分类结果-混淆矩阵的四个元素
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

        returns:
            hits, misses, falsealarms, correctnegatives
            #aliases: TP, FN, FP, TN 
        '''
        #根据阈值分类为 0, 1
        self.o = np.where(self.o >= threshold, 1, 0)
        self.s = np.where(self.s >= threshold, 1, 0)

        # True positive (TP)
        hits = np.sum((self.o == 1) & (self.s == 1))

        # False negative (FN)
        misses = np.sum((self.o == 1) & (self.s == 0))

        # False positive (FP)
        falsealarms = np.sum((self.o == 0) & (self.s == 1))

        # True negative (TN)
        correctnegatives = np.sum((self.o == 0) & (self.s == 0))

        return hits, misses, falsealarms, correctnegatives

    def precision(self,threshold=0.1):
        '''
        func: 计算精确度precision: TP / (TP + FP)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

        returns:
            dtype: float
        '''

        TP, FN, FP, TN = self.prep_clf(threshold=threshold)

        return TP / (TP + FP)

    def recall(self, threshold=0.1):
        '''
        func: 计算召回率recall: TP / (TP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

        returns:
            dtype: float
        '''

        TP, FN, FP, TN = self.prep_clf(threshold=threshold)

        return TP / (TP + FN)

    def ACC(self, threshold=0.1):
        '''
        func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

        returns:
            dtype: float
        '''

        TP, FN, FP, TN = self.prep_clf(threshold=threshold)

        return (TP + TN) / (TP + TN + FP + FN)

    def FSC(self, threshold=0.1):
        '''
        func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
        '''
        precision_socre = self.precision(threshold=threshold)
        recall_score = self.recall(threshold=threshold)

        return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))

