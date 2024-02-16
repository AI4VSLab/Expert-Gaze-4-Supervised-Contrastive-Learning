import sys
sys.path.append("/../../Self-Supervised-Learning/")

import pytorch_lightning as pl
from dataset.dataset.eye_tracking_dataset import EyeTracking_Dataset
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import torch

from dataset.utils.sampler import *
from torch.utils.data import DataLoader 

from dataset.datamodule.ReportDataModule import expertise, EXPERTS, ReportDataModule

# make sure to go from 0 to max num special token continously, because this is taken advantage in making the mask later 
SPECIAL_TOKENS = {
                'pad_index':0,
                'cls_index':1,
                'eos_index':2
}

# we also denote where the current fixation is in terms of region in A B C ..., this is a mapping to index
REGION_MAP = {
                'A': 0,
                'B': 1,
                'C': 2,
                'D': 3,
                'E': 4,
                'F': 5,
                'G': 6,
                'H': 7,
                'I': 8,
                'K': 9,
                'Circumpapillary_RNFL': 0, 
                'En-face_52.0micrometer_Slab_(Retina_View)': 1, 
                'RNFL_Thickness_(Retina_View)': 2, 
                'RNFL_Probability_and_VF_Test_points(Field_View)': 3, 
                'GCL+_Thickness_(Retina_View)': 4, 
                'GCL+_Probability_and_VF_Test_points': 5
}





class EyeTrackingDM(pl.LightningDataModule):
    
    def __init__(self,
                 mode: str = 'ssl',
                 split: float = 0.8,
                 split_trainset: float = 1.0,
                 path_eyeTrack: str = None,
                 path_pavlovia: str = None,
                 path_tobii: str = None,
                 path_report: str = None,
                 max_seq_length = 2000,
                 batch_size = 16,
                 posPairs_criteria = 'experimentANDfeedback',
                 dataset_expert_only = False,
                linear_eval: bool = False
                 ) -> None:
        super().__init__()

        self.mode = mode
        self.split = split
        self.split_trainset = split_trainset
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.posPairs_criteria = posPairs_criteria
        self.dataset_expert_only = dataset_expert_only
        self.split_trainset = split_trainset
        self.path_pavlovia, self.path_tobii, self.path_report = path_pavlovia, path_tobii, path_report
        self.train_split_idx = None


        self.df_label = pd.read_csv(f'{path_eyeTrack}/label.csv').set_index('report')

        # ============================================= grab all data and labels =============================================
        self.gaze_data, vocab, feedback = EyeTrackingDM.get_data(path_eyeTrack, False, df_label = self.df_label, path_pavlovia = path_pavlovia, path_tobii = path_tobii , dataset_expert_only = dataset_expert_only)
        
       
        self.fixName_score, self.fixName_notValid = feedback
        # vocab[word/idx] = index in vocab, ie vocab[cls_index] = 1 as shown above
        self.vocab_fix, self.vocab_fix_region, self.vocab_region = vocab
        # ============================================= choose which data to use =============================================
        #self.seq = list(self.gaze_data['seq_region']) 
        #self.labels, self.labels_expert = np.array(self.labels), np.array(self.labels_expert)

        # ============================================= preprocess =============================================
        # pad and replace the column with padded values
        self.gaze_data['seq_region'] = self.pad_inputs(list(self.gaze_data['seq_region']) ) 
        
        self.fixName_feedbackClass = self.preprocess_feedback(self.fixName_score)

        # make sure different gaze on same image arent in both train and test set
        # split first reports trainset first, then get positive pair -> so dont get positive pair from test in trainset
        # split by report first, by calling report dm method
      
        # initialize variables
        self.train_split_idx = None
        self.train_idx = self.test_idx = self.r2f_train = self.r2f_test = self.reports_train = self.reports_test = []

        self.save_hyperparameters(logger= False)
    
    def configure_mode(self, mode):
        self.mode = mode


    #####################################################################################################################################
    #                                                  Setting Up Code
    #####################################################################################################################################
    @staticmethod
    def get_powersets():
        '''
        make a set of letters ie A, AA, AAA, AB to map idx to a given letter. Used by SGT
        '''
        from itertools import chain, combinations
        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        
        let_power_set = ['AA']+list(powerset("ABCDEFGHI"))[1:]
        mapping = {i: ''.join(list(let_power_set[i]))  for i in range(len(let_power_set))}

        # changed from ABCDEFGH
        let_power_set_region = ['AA']+list(powerset("ABCDEFGHIJ"))[1:]
        mapping_region = {i: ''.join(list(let_power_set_region[i]))  for i in range(len(let_power_set_region))}

        return mapping, mapping_region

    @staticmethod
    def get_data(
                path, 
                sgt = False,
                df_label = None, 
                path_pavlovia = None,
                path_tobii = None,
                dataset_expert_only = False
                ):
        '''
        Both actual eyetrack data and feedback functions return fixation name used, 
        we will only use the ones that are both, to guarantee they are in both, will first 
        get all the fix from feedback. and that list to get eyetrack data
        '''
        # first get the ones in feedbback
        fixName_score, fixName_notValid = EyeTrackingDM.get_data_feedback(path)

        # now only if fixation is in feedback, then we will get them
        gaze_data, vocab = EyeTrackingDM.get_data_eyetrack(
            path, 
            validFix= {n: True for n in fixName_score} ,
            df_label = df_label, 
            path_pavlovia = path_pavlovia,
            path_tobii = path_tobii,
            dataset_expert_only = dataset_expert_only
            )        

        # ============================================= repack outputs =============================================
        feedback = (fixName_score, fixName_notValid)
        return gaze_data, vocab, feedback

    @staticmethod
    def get_data_feedback(path):
        df_feedback = pd.read_csv(f'{path}/feedback.csv')
        fixName_score = {}
        fixName_notValid = {}
        for i, row in df_feedback.iterrows():
            if np.isnan(row.score): fixName_notValid[row.FixName] = True
            else: fixName_score[row.FixName] = int(row.score)
        return fixName_score, fixName_notValid            


    @staticmethod
    def get_data_eyetrack(
                path, 
                validFix = None,
                df_label = None, 
                path_pavlovia = None,
                path_tobii = None,
                dataset_expert_only = False
                ):
        '''
        @params
            path: path to eyetracking data folder
            sgt: if we are getting outputs for sgt
            validFix: dict[validFix] = True, only fixaitons in this dict are used. else we don't. we don't use them because
            they might not have a comment from participant. If None, return all reports
        @returns:
            idx_currFixName: 
        '''
        # ============================================= get powerset =============================================
        mapping, mapping_region = EyeTrackingDM.get_powersets()

        # ============================================= get rpt to fix mappings =============================================
        report_pavloviaFixation, pavloviaFixation_report = ReportDataModule.get_pavlovia_fixation_paths(path_pavlovia)
        report_tobiiFixation, tobiiFixation_report = ReportDataModule.get_tobii_fixation_paths(path_tobii)

        # ============================================= get all subdir =============================================
        items = os.listdir(path)
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

        # ============================================= get vocab size =============================================
        vocab_fix_region, vocab_region, vocab_fix = SPECIAL_TOKENS.copy(), SPECIAL_TOKENS.copy(), SPECIAL_TOKENS.copy() # start with these dicts

        # ============================================= definte data to extract =============================================
        currFixName_idx = {}
        
        # data
        seq_letters = [] # fixation but letters ie AA, for SGT
        seq_region_letters = [] # same as below but with letter
        seq_region = [] # fixation split into regions then each region into grids in class idx
        region = [] # fixation, each fixation is assigned to a region, ie A/B/C...
        # label
        labels = [] # Glaucoma/Healthy
        allCurrFixName = [] # name of experiment+fixation
        currExpName = [] # EEM16 for ex
        labels_expert = [] # 1.0 for expert, 0.0 for novice
    
        counter = 0
        for folder in folders: # folder ie EEM18 or EEM1_fixations
            folder_path = os.path.join(path, folder) 
            csvs = os.listdir(folder_path) 
            # ============================================ read expertise and skip if get expert only============================================
            # skip EEM13 since it was not well calibrated
            if 'EEM13' in folder: 
                print(f'We are skipping EEM13 {folder}')
                continue
            if not( expertise[folder.split('_')[0]] in EXPERTS ) and dataset_expert_only: continue # skip this current experiment if only want expert
            # ============================================ read through csvs ============================================
            for i,csv in enumerate(csvs):
                # ============================================ read csv and skip if not valid fixation ============================================
                csv_ = csv.removesuffix('.csv')
                if validFix and csv_ not in validFix: continue
                # read csv
                csv_path = os.path.join(folder_path, csv)
                df = pd.read_csv(csv_path)      

                # NOTE: skip the ones with region_word_idx that is 0
                df_copy = df.dropna()      
                if len(list(df_copy.region_word_idx)) == 0 : 
                    print(f'We are skipping this: {csv_}')
                    continue

                # ============================================ read expertise ============================================
                if not( expertise[folder.split('_')[0]] in EXPERTS ): labels_expert.append( 0.0 )
                else: labels_expert.append( 1.0 )
                # ============================================ read grid on entire image ============================================
                seq_letters.append( [  mapping[idx] for idx in  list(df.word_idx)  ])
                for idx in list(df.word_idx): vocab_fix[idx] = int(idx) + len(SPECIAL_TOKENS)
                # ============================================ read grid on from each region ============================================
                df = df.dropna()
                seq_region_letters.append( [  mapping_region[idx] for idx in  list(df.region_word_idx)  ] )
                seq_region.append(  list(df.region_word_idx) )
                # ============================================ read which region ============================================
                region.append(  list(df.region) )
                
                # ============================================ read label ============================================
                # ; check pav and tobi dict and see which one this curr fix is in 
                l = None
                if pavloviaFixation_report[csv_] and report_tobiiFixation[csv_]: l = pavloviaFixation_report[csv_]
                elif pavloviaFixation_report[csv_]: l = pavloviaFixation_report[csv_]
                elif tobiiFixation_report[csv_]: l = tobiiFixation_report[csv_]
                else: print(csv_, l)
                labels.append(1 if df_label.loc[l].label == 'Glaucoma' else 0)

                # ============================================ read name of current fixation and experient name ============================================
                allCurrFixName.append( csv_)#'_'.join( [folder.split('_')[0], csv] ) )
                currExpName.append(folder)
                # ============================================ count vocab ============================================
                for idx in list(df.region_word_idx): vocab_fix_region[idx] = int(idx) + len(SPECIAL_TOKENS)
                for r in list(df.region): vocab_region[idx] = REGION_MAP[r] + len(SPECIAL_TOKENS)
                
                # ============================================ create idx map ============================================
                currFixName_idx[csv_] = counter
                counter += 1

        gaze_data = pd.DataFrame(
                        {
                        'idx': [i for i in range(len(allCurrFixName))],
                        'gaze_name': allCurrFixName,  
                        'ExpName': currExpName,  
                        'seq_letters': seq_letters,
                        'seq_region_letters': seq_region_letters,
                        'seq_region': seq_region,
                        'region': region,
                        'labels': labels,
                        'labels_expert': labels_expert
                        })
        gaze_data = gaze_data.set_index('gaze_name')
        print(f'There are {len(seq_letters)} number of fixations')
        vocabs = (vocab_fix, vocab_fix_region, vocab_region)
        return gaze_data, vocabs


    #####################################################################################################################################
    #                                                       Data Preprocessing
    #####################################################################################################################################
    @staticmethod
    def train_test_split(
                        split,
                        gaze_data,
                        path_pavlovia, 
                        path_tobii,
                        path_report,
                        train_idx = [],
                        test_idx = [],
                        r2f_train = [],
                        r2f_test = [],
                        reports_train = [],
                        reports_test = []
                        ):
        '''if train_idx and test_idx provided, then don't sample -> used in loading back ckpt'''
        # =================================================== step 1: sample training reports  ===================================================
        # should lowkey make the training idx and other into an object
        if not train_idx and not test_idx and not r2f_train and not r2f_test and not reports_train and not reports_test:
            # grab inputs needed for get_split_reports_fixation_pair
            report_pavloviaFixation, pavloviaFixation_report = ReportDataModule.get_pavlovia_fixation_paths(path_pavlovia)
            report_tobiiFixation, tobiiFixation_report = ReportDataModule.get_tobii_fixation_paths(path_tobii)
            # get reports list
            _, report_path, report_label = ReportDataModule.get_report_paths(path_report)
            reports = np.array([k for k in report_path])
            # call split fcuntion
            train_idx,test_idx, r2f_train, r2f_test, reports_train, reports_test = ReportDataModule.get_split_simple(split , reports, report_tobiiFixation, report_pavloviaFixation, reports_train, reports_test, gaze_data )
        
        # =================================================== step 2: select the fixations corrspound to sampled trainiang reports  ===================================================
        allCurrFixName_train, allCurrFixName_test = [], []
        r_used = list(gaze_data.index)
        for k in r2f_train: 
            if  len(r2f_train[k])==0: continue
            f_filtered = [f for f in r2f_train[k] if f in  r_used] # we might get assigned a fixation, but we not want it since it is not valid
            allCurrFixName_train.extend( f_filtered )
        for k in r2f_test: 
            if  len(r2f_test[k])==0: continue
            f_filtered = [f for f in r2f_test[k] if f in  r_used]
            allCurrFixName_test.extend( f_filtered)

        seq_train, seq_test = [], []
        for fs in allCurrFixName_train: 
            seq_train.append(   gaze_data.loc[fs].seq_region   )
        for fs in allCurrFixName_test: 
            seq_test.append( gaze_data.loc[fs].seq_region  )
        
        return np.array(seq_train), np.array(seq_test), train_idx, test_idx, r2f_train, r2f_test, allCurrFixName_train, allCurrFixName_test, reports_train, reports_test

    
    def pad_inputs(self, data):
        # fix[1] + 2 to offset for pad_index and cls_index
        data = [ np.concatenate( ( [ SPECIAL_TOKENS['cls_index'] ] ,np.array(fix)+ len(SPECIAL_TOKENS) , [ SPECIAL_TOKENS['eos_index'] ] ,[0] * (self.max_seq_length - len(fix))) )  for fix in  data ] 
        # trim, notice if sequence longer than max_seq_length we might through away eos as well, consistent with BERT
        data =  [d[:self.max_seq_length] if len(d) > self.max_seq_length else d for d in data]
        return data

    def preprocess_feedback(self, fixName_score):
        fixName_feedbackClass = {}
        for fixName in fixName_score:
            fixName_feedbackClass[fixName] = 1.0 if fixName_score[fixName] >= 50 else 0.0
        return fixName_feedbackClass

    
    @staticmethod
    def get_positive_pairs_experiments(sequence, seqExpLabel):
        '''
        @params:
            seqExpLabel: a list seqExpLabel[i] = 'EEM16'/'EEM16_control' for example 
        @returns:
            []: a list of index, mapping in Expidx, mapping each folder to an index
        '''
        Expidx = {}
        idx = 0
        for exp in seqExpLabel: 
            if exp not in Expidx: 
                Expidx[exp] = idx
                idx += 1
        return  np.array([ Expidx[exp]  for exp in seqExpLabel]), Expidx

    @staticmethod
    def get_positive_pairs_expFeedback(sequence, criteria_label):
        '''
        decide which ones are not only from the same expriment, but also given the same classification (not groud truth)
        if so, assign same index

        NOTE: here we bin EEM9 and EEM9_control as the same experiment/persion
        
        @params
            criteria_label: contains
                currExpName: list of which current fixation belong to, ie currExpName[ allCurrFixName_train[i] ] = EEM16 
                allCurrFixName: list of name of current fix ie 'EEM1_9193_OS_2021_widefield_report'
                fixName_feedbackClass: dict[fix name ] = 1.0/0.0 classified during experiment 
        @returns
            currFixFeedbackClass: list, currFixFeedbackClass[i] is the class from participant of allCurrFixName_train[i]
        '''
        currExpName, allCurrFixName, fixName_feedbackClass = criteria_label
        assert len(currExpName) == len(allCurrFixName)
        
        currFixFeedbackClass = []

        expANDclass = {}
        num_class_name = 0

        # build a dict that maps ie 'EEM1_1.0' to idx, and use that index to assign class to each fixation 
        for i,currFixName in enumerate(allCurrFixName):
            pred = fixName_feedbackClass[currFixName] # pred from participant
            class_name = f'{currExpName[i].split("_")[0] }_{str(pred)}'
            if class_name not in expANDclass: 
                expANDclass[class_name] = num_class_name
                num_class_name += 1
            currFixFeedbackClass.append( expANDclass[class_name] )

        return np.array(currFixFeedbackClass), expANDclass
    

    @staticmethod
    def get_positive_pairs_Feedback(sequence, criteria_label):
        '''
        decide which only from given the same classification (not groud truth) if so, assign same index

        NOTE: here we bin EEM9 and EEM9_control as the same experiment/persion
        @params
            criteria_label: contains
                currExpName: list of which current fixation belong to, ie currExpName[ allCurrFixName_train[i] ] = EEM16 
                allCurrFixName: list of name of current fix ie 
                fixName_feedbackClass: dict[fix name ] = 1.0/0.0 classified during experiment 
        @returns
            currFixFeedbackClass: list, currFixFeedbackClass[i] is the class from participant of allCurrFixName_train[i]
        '''
        allCurrFixName, fixName_feedbackClass = criteria_label
        
        
        currFixClass = []
        classOnly = {}
        num_class_name = 0

        # build a dict that maps ie 'EEM1_1.0' to idx, and use that index to assign class to each fixation 
        for i,currFixName in enumerate(allCurrFixName):
            pred = fixName_feedbackClass[currFixName] # pred from participant
            class_name = str(pred)
            if class_name not in classOnly: 
                classOnly[class_name] = num_class_name
                num_class_name += 1
            currFixClass.append( classOnly[class_name] )

        return np.array(currFixClass), classOnly

    def get_positive_pairs(self, sequence, criteria = 'experiment'):
        '''
        create a mask for what other sequnces are positive paris
        @params
            sequence: data
            # legacy: criteria_label: whatever infomration needed by the criteria
        '''
        if criteria == 'experiment': 
            return EyeTrackingDM.get_positive_pairs_experiments(sequence, self.currExpName[self.train_idx])
        elif criteria == 'experimentANDfeedback': 
            criteria_label = (list(self.gaze_data.loc[self.allCurrFixName_train].ExpName) , self.allCurrFixName_train, self.fixName_feedbackClass)
            return EyeTrackingDM.get_positive_pairs_expFeedback(sequence, criteria_label)
        elif criteria == 'feedback':
            criteria_label = ( self.allCurrFixName_train, self.fixName_feedbackClass)
            return EyeTrackingDM.get_positive_pairs_Feedback(sequence, criteria_label)

    

    #####################################################################################################################################
    #                                                       Data Module Methods 
    #####################################################################################################################################
    def setup(self, 
              stage: str = ''
        ):

        # do this again, since we might be loading from ckpt
        self.seq_train, self.seq_test, self.train_idx, self.test_idx, self.r2f_train, self.r2f_test, self.allCurrFixName_train, self.allCurrFixName_test, self.reports_train, self.reports_test = EyeTrackingDM.train_test_split(
            self.split,
            self.gaze_data,
            self.path_pavlovia, 
            self.path_tobii,
            self.path_report,
            train_idx = self.train_idx,
            test_idx = self.test_idx,
            r2f_train = self.r2f_train,
            r2f_test = self.r2f_test,
            reports_train = self.reports_train,
            reports_test = self.reports_test
        )

        self.labels_postive, self.labels_postive_map =  self.get_positive_pairs(self.seq_train  , criteria = self.posPairs_criteria)
        
        labels_train = self.gaze_data.loc[self.allCurrFixName_train].labels.to_numpy()
        labels_test = self.gaze_data.loc[self.allCurrFixName_test].labels.to_numpy()

        labels_expertise_train = self.gaze_data.loc[self.allCurrFixName_train].labels_expert.to_numpy()
        labels_expertise_test = self.gaze_data.loc[self.allCurrFixName_test].labels_expert.to_numpy()

        labels_feedback_train = []
        labels_feedback_test = []
        for currFix in  list(self.gaze_data.loc[self.allCurrFixName_train].index):
            labels_feedback_train.append(self.fixName_feedbackClass[currFix])
        for currFix in  list(self.gaze_data.loc[self.allCurrFixName_test].index):
            labels_feedback_test.append(self.fixName_feedbackClass[currFix])
        labels_feedback_train, labels_feedback_test = np.array(labels_feedback_train), np.array(labels_feedback_test)



        assert len(labels_feedback_train) != 0 and len(labels_feedback_test) != 0 
        # ============================================ trainset split ============================================
        if not self.train_split_idx:
            idx  = torch.randperm(len(self.seq_train)) 
            self.train_split_idx = idx[:int(self.split_trainset*len(self.seq_train))].tolist()


        self.trainset = EyeTracking_Dataset(
                        sequence= self.seq_train[self.train_split_idx],
                        labels = labels_train[self.train_split_idx],
                        labels_expert= labels_expertise_train[self.train_split_idx],
                        labels_feedback= labels_feedback_train[self.train_split_idx], 
                        mode = self.mode,
                        labels_postive = self.labels_postive[self.train_split_idx],
                        allCurrFixName = np.array(self.allCurrFixName_train)[self.train_split_idx]
        )

    
        self.trainset = EyeTracking_Dataset(
                        sequence= self.seq_train[self.train_split_idx],
                        labels = labels_train[self.train_split_idx],
                        labels_expert= labels_expertise_train[self.train_split_idx],
                        labels_feedback= labels_feedback_train[self.train_split_idx], 
                        mode = self.mode,
                        labels_postive = self.labels_postive[self.train_split_idx]
        )
        
        self.testset = EyeTracking_Dataset(
                        sequence= self.seq_test,
                        labels = labels_test,
                        labels_expert= labels_expertise_test,
                        labels_feedback=labels_feedback_test, 
                        mode = 'test',
                        labels_postive = self.labels_postive,
                        allCurrFixName = self.allCurrFixName_test
        )

        # for now use trainset as val test
        self.valset = self.trainset

        return

    def train_dataloader(self, num_workers  = 32):
        labels_train = self.gaze_data.loc[self.allCurrFixName_train].labels.to_numpy()[self.train_split_idx]
        dl =   DataLoader(
                            self.trainset, 
                            batch_size = self.batch_size,
                            sampler = get_sampler( labels_train ) if self.mode == 'sl' else None,
                            num_workers = num_workers
                )
        return dl
        
    
    def val_dataloader(self, num_workers = 32):
        return DataLoader(
                            self.trainset, 
                            batch_size = self.batch_size,
                            num_workers = num_workers
                )

    def test_dataloader(self):
        return DataLoader(
                    self.testset, 
                    batch_size = self.batch_size
                    )
    

    def teardown(self, stage: str):
        return

    def state_dict(self):
        # called when we do a checkpoint, ckpt made by the trainer
        state = {
                'train_idx': self.train_idx,
                'test_idx': self.test_idx,
                'reports_train': self.reports_train,
                'reports_test': self.reports_test,
                'allCurrFixName_train': self.allCurrFixName_train,
                'allCurrFixName_test': self.allCurrFixName_test,
                'r2f_train': self.r2f_train,
                'r2f_test': self.r2f_test,
                'train_split_idx': self.train_split_idx,
                'r2f_test': self.r2f_test,
                'train_split_idx': self.train_split_idx
                }
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.train_idx, self.test_idx = state_dict['train_idx'], state_dict['test_idx']
        self.reports_train, self.reports_test = state_dict['reports_train'], state_dict['reports_test']
        self.allCurrFixName_train, self.allCurrFixName_test = state_dict['allCurrFixName_train'], state_dict['allCurrFixName_test']
        self.r2f_train, self.r2f_test = state_dict['r2f_train'], state_dict['r2f_test']
        self.train_split_idx = state_dict['train_split_idx']

      
    def saveEmbeddings(self, features_train, features_test,path = 'mat.npz'):
        np.savez(path, features_train=features_train, 
                        features_test = features_test,
                        allCurrFixName_train= self.allCurrFixName_train,
                        allCurrFixName_test = self.allCurrFixName_test,
                        reports_train = self.reports_train,
                        reports_test = self.reports_test,
                        train_idx = self.train_idx,
                        test_idx = self.test_idx,
                        r2f_train = self.r2f_train,
                        r2f_test = self.r2f_test,
                        train_split_idx = self.train_split_idx, # split within trainset
                        allow_pickle = True
                        )

    def load_embeddings(self, path):
        ftype = path.split('.')[-1]
        if ftype == 'npz':
            # assume their names are the following 
            Mnpz = np.load(path, allow_pickle= True)
            self.features_train = Mnpz['features_train']
            self.features_test = Mnpz['features_test']
            self.allCurrFixName_train= Mnpz['allCurrFixName_train'] 
            self.allCurrFixName_test = Mnpz['allCurrFixName_test'] 
            self.reports_train = Mnpz['reports_train'] 
            self.reports_test = Mnpz['reports_test'] 
            self.train_idx = Mnpz['train_idx'] 
            self.test_idx =  Mnpz['test_idx'] 
            self.r2f_train = Mnpz['r2f_train'] 
            self.r2f_test = Mnpz['r2f_test'] 
            self.train_split_idx = Mnpz['train_split_idx']


        elif ftype == 'ckpt':
            pass
