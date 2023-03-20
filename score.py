# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
from typing import Union
import math
from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
import jsonlines
import torch
import multiprocessing
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
from nltk import sent_tokenize
from tqdm import tqdm
import numpy as np
import pandas as pd
# data utils
def write_predict(write_path, data, eval_scores):
    for i in range(len(data)):
        data[i]['predict_scores'] = { "coherence":  eval_scores[i], "consistency":  eval_scores[i],  "fluency":  eval_scores[i], "relevance":  eval_scores[i],  "overall":eval_scores[i]}
    with open(write_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        print('\nPredicted scores are saved in {}'.format(write_path))
            
def load_json(data_path):
    with open(data_path) as f:
        data = json.loads(f.read())
    return data

# calculate correlations
def calculate_correlation(pred_score, human_score, dim, result):
    assert len(pred_score) == len(human_score)
    if dim not in result:
        result[dim] = [0] * 3
    result[dim][0] += pearsonr(pred_score, human_score)[0]
    result[dim][1] += spearmanr(pred_score, human_score)[0]
    result[dim][2] += kendalltau(pred_score, human_score)[0]
    return result

def print_correlations(result,output_url):
    table = PrettyTable(['Dimensions','Pearson', 'Spearman', 'Kendall'])
    output=pd.DataFrame(columns=['Dimensions','Pearson', 'Spearman', 'Kendall'])
    index=0
    for dim in result:
        table.add_row([dim, round(result[dim][0], 6), round(result[dim][1], 6),   round(result[dim][2], 6)])
        output.loc[index]=[dim, round(result[dim][0], 6), round(result[dim][1], 6),   round(result[dim][2], 6)]
        index+=1
    print(table)
    output.to_csv(output_url+'.csv')
    
    
    

def get_unique_value(data, key):
    value = set()
    for i in range(len(data)):
        if data[i][key] not in value:
            value.add(data[i][key])
    return list(value)

def correlation_for_summ(data, overall, output_url):
    dimensions = ['coherence', 'consistency', 'fluency', 'relevance']
    if overall == True:
        dimensions.append('overall')

    """
    # sample level correlation
    print('\n ********** Sample Level Correlations *********')
    result = {}
    for dim in dimensions:
        pred_score, human_score = [], []
        for i in range(len(data)):
            pred_score.append(data[i]['predict_scores'][dim])
            human_score.append(data[i]['scores'][dim])
        result = calculate_correlation(pred_score, human_score, dim, result)
    print_correlations(result)
    """
    
    
    # summary level correlation
    print('\n ********* Summary Level Correlations *********')
    result = {}
    docs = get_unique_value(data, 'doc_id')
    for dim in dimensions:
        valid_cnt = 0
        for doc_idx in docs:
            pred_score, human_score = [], []
            for i in range(len(data)):
                if data[i]['doc_id'] == doc_idx:
                    pred_score.append(data[i]['predict_scores'][dim])
                    human_score.append(data[i]['scores'][dim])
            if len(set(pred_score)) == 1 or len(set(human_score)) == 1:
                continue
            result = calculate_correlation(pred_score, human_score, dim, result)
            valid_cnt += 1
        for j in range(3):
            result[dim][j] /= valid_cnt
    print_correlations(result,output_url)
    




    
    """            
    # system level correlations
    print('\n ********** System Level Correlations *********')
    result = {}
    systems = get_unique_value(data, 'system_id')
    for dim in dimensions:
        pred_score, human_score = [], []
        for system_idx in systems:
            doc_cnt = 0
            cur_pred, cur_human = 0, 0
            for i in range(len(data)):
                if data[i]['system_id'] == system_idx:
                    cur_pred += data[i]['predict_scores'][dim]
                    cur_human += data[i]['scores'][dim]
                    doc_cnt += 1
            pred_score.append(cur_pred / doc_cnt)
            human_score.append(cur_human / doc_cnt)
        result = calculate_correlation(pred_score, human_score, dim, result)
    print_correlations(result)
    """
    
      

def main() -> None:

    # set url
    seed_everything(12)
    data_path="./data/Test.jsonl"
    

    checkpoints_url="./lightning_logs/version_51604/checkpoints/" #doc-sum-ref
    hparams_url="./lightning_logs/version_51604/hparams.yaml" #doc-sum-ref
    output_url='./newest_sentence/'
    checkpoints_list=os.listdir(checkpoints_url)
  
 
    for checkpoints in checkpoints_list:
        cur_checkpoints_url=checkpoints_url+checkpoints
        print(cur_checkpoints_url)
        # load model 
        model = load_from_checkpoint(cur_checkpoints_url, hparams_url)
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        print(model.device)
        print('[Done] Load Model Successfully!')
        
        # load data
        human_annotation = {}
        with open(data_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                pair_id = item['id'] + "-" + item['model_id']
    
                human_annotation[pair_id] = {
                    'doc_id':item['id'],
                    'system_id':item['model_id'],
                    'source': item['text'],
                    'reference': item['references'][0],
                    'system_output': item['decoded'],
                }
                avg_expert = {'coherence': 0, 'consistency': 0, 'fluency': 0, 'relevance': 0}
                for expert in item['expert_annotations']:
                    for k, v in expert.items():
                        avg_expert[k] += v / len(item['expert_annotations'])
                        
                overall=0
                for k in avg_expert.keys():
                    overall+=avg_expert[k]/4
                avg_expert['overall']=overall
                human_annotation[pair_id]['scores']=avg_expert
        
        dataloader = DataLoader(
            dataset=list(human_annotation.values()),
            batch_size=4,
            num_workers=multiprocessing.cpu_count(),
        )
        Softmax=nn.Softmax()
    
    
        
        # predit scores (whole summary)
        print('whole summary')
        task_list = ['hyp-ref','hyp-src','hyp-src-ref']
        for input_segments in task_list:
            eval_scores=[]
            for batch_input in tqdm(dataloader):
                
                src_inputs = model.encoder.prepare_sample(batch_input["source"])
                ref_inputs = model.encoder.prepare_sample(batch_input["reference"])
                mt_inputs = model.encoder.prepare_sample(batch_input["system_output"])
                
                
                src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
                ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
                mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
                
                inputs = {**src_inputs, **mt_inputs, **ref_inputs}
                for k in inputs.keys():
                    inputs[k]=inputs[k].to(device)
                    
                batch_prediction = model.forward(input_segments=input_segments, **inputs)
                scores=Softmax(batch_prediction['score'])[:,1].view(-1).tolist()
                
                eval_scores.extend(scores)
    
            write_predict(output_url+checkpoints+input_segments+'.json', list(human_annotation.values()), eval_scores)
            data = load_json(output_url+checkpoints+input_segments+'.json')
            correlation_for_summ(data,True,output_url+checkpoints+input_segments)
    
   
    for checkpoints in checkpoints_list:
        cur_checkpoints_url=checkpoints_url+checkpoints
        print(cur_checkpoints_url)
        # load model 
        model = load_from_checkpoint(cur_checkpoints_url, hparams_url)
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        print(model.device)
        print('[Done] Load Model Successfully!')
        
        # load data
        human_annotation = {}
        with open(data_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                pair_id = item['id'] + "-" + item['model_id']
    
                human_annotation[pair_id] = {
                    'doc_id':item['id'],
                    'system_id':item['model_id'],
                    'source': item['text'],
                    'reference': item['references'][0],
                    'system_output': item['decoded'],
                }
                avg_expert = {'coherence': 0, 'consistency': 0, 'fluency': 0, 'relevance': 0}
                for expert in item['expert_annotations']:
                    for k, v in expert.items():
                        avg_expert[k] += v / len(item['expert_annotations'])
                        
                overall=0
                for k in avg_expert.keys():
                    overall+=avg_expert[k]/4
                avg_expert['overall']=overall
                human_annotation[pair_id]['scores']=avg_expert
        
        dataloader = DataLoader(
            dataset=list(human_annotation.values()),
            batch_size=1,
            num_workers=multiprocessing.cpu_count(),
        )
        Softmax=nn.Softmax()
    
    
        # predit scores (sentence by sentence)
        print('sentence-level')
        task_list = ['hyp-ref']
        for input_segments in task_list:
            final_eval_scores1=[]
            for batch_input in tqdm(dataloader):
                sent_list=sent_tokenize(batch_input["system_output"][0])
                print(sent_list)
                src_inputs = model.encoder.prepare_sample(batch_input["source"]*len(sent_list))
                ref_inputs = model.encoder.prepare_sample(batch_input["reference"]*len(sent_list))
                mt_inputs = model.encoder.prepare_sample(sent_list)
                
                
                src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
                ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
                mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
                
                inputs = {**src_inputs, **mt_inputs, **ref_inputs}
                for k in inputs.keys():
                    inputs[k]=inputs[k].to(device)
                    
                batch_prediction = model.forward(input_segments=input_segments, **inputs)
                scores=Softmax(batch_prediction['score'])[:,1].view(-1).tolist()
                final_eval_scores1.append(np.mean(scores))
            write_predict(output_url+checkpoints+input_segments+'.json', list(human_annotation.values()), final_eval_scores1)
            data = load_json(output_url+checkpoints+input_segments+'.json')
            correlation_for_summ(data,True,output_url+checkpoints+input_segments)
   
                

        print('sentence-level')
        task_list = ['hyp-src']
        for input_segments in task_list:
            final_eval_scores2=[]
            for batch_input in tqdm(dataloader):
                sent_list=sent_tokenize(batch_input["system_output"][0])
   
                src_inputs = model.encoder.prepare_sample(batch_input["source"]*len(sent_list))
                ref_inputs = model.encoder.prepare_sample(batch_input["reference"]*len(sent_list))
                mt_inputs = model.encoder.prepare_sample(sent_list)
                
                
                src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
                ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
                mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
                
                inputs = {**src_inputs, **mt_inputs, **ref_inputs}
                for k in inputs.keys():
                    inputs[k]=inputs[k].to(device)
                
                    
                batch_prediction = model.forward(input_segments=input_segments, **inputs)
                scores=Softmax(batch_prediction['score'])[:,1].view(-1).tolist()
                final_eval_scores2.append(np.mean(scores))
                
            write_predict(output_url+checkpoints+input_segments+'.json', list(human_annotation.values()), final_eval_scores2)
            data = load_json(output_url+checkpoints+input_segments+'.json')
            correlation_for_summ(data,True,output_url+checkpoints+input_segments)
            
            
                        
  
        combine=[]
        for i in range(len(final_eval_scores1)):
            combine.append(math.sqrt(final_eval_scores1[i]*final_eval_scores2[i]))
    
        write_predict(output_url+'result_gavg.json', list(human_annotation.values()), combine)
        data = load_json(output_url+'result_gavg.json')
        correlation_for_summ(data,True,output_url+'result_gavg')
        
        combine=[]
        for i in range(len(final_eval_scores1)):
            combine.append((final_eval_scores1[i]+final_eval_scores2[i])/2)
    
        write_predict(output_url+'result_avg.json', list(human_annotation.values()), combine)
        data = load_json(output_url+'result_avg.json')
        correlation_for_summ(data,True,output_url+'result_avg')
        
        combine=[]
        for i in range(len(final_eval_scores1)):
            combine.append(min(final_eval_scores1[i],final_eval_scores2[i]))
    
        write_predict(output_url+'result_min.json', list(human_annotation.values()), combine)
        data = load_json(output_url+'result_min.json')
        correlation_for_summ(data,True,output_url+'result_min')
        
        combine=[]
        for i in range(len(final_eval_scores1)):
            combine.append(max(final_eval_scores1[i],final_eval_scores2[i]))
    
        write_predict(output_url+'result_max.json', list(human_annotation.values()), combine)
        data = load_json(output_url+'result_max.json')
        correlation_for_summ(data,True,output_url+'result_max')
       

            


if __name__ == '__main__':
    main()
