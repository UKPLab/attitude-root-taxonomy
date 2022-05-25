import json
import glob

MODES = ["st_baseline_full",
        "st_baseline_collapse",
        "transformer_baseline_full",
        "transformer_baseline_collapse",
        "setfit_baseline_full",
        "setfit_baseline_collapse",
        "transformer_zero_full",
        "transformer_zero_collapse",
        "setfit_zero_full",
        "setfit_zero_collapse",       
        "transformer_pretrain_full",
        "transformer_pretrain_collapse",      
        "use_ft_transformer_full",
        "use_ft_transformer_collapse",
        "setfit_pretrain_full",
        "setfit_pretrain_collapse",
        ]
path = "output/*_test_results.json"

for file in glob.glob(path):
    if 'transformer_pretrain' in file:
        continue
    if 'full' in file:
        num_roots = 11
    elif 'collapse' in file:
        num_roots = 7  
    for mode in MODES:
        if mode in file:            
            with open(file, 'r') as f:
                for line in f:
                    results = dict(json.loads(line))
                    f1 = round(results["mac_mean"],2)
            if 'st_baseline' in mode:
                print('Sentence Transformer and Logistic Regression baseline in {} root scenario:'.format(num_roots))
            
            elif 'transformer_baseline' in mode:
                print('RoBERTa-base standard fine-tuning in {} root scenario:'.format(num_roots))
            
            elif 'setfit_baseline' in mode:
                print('SetFit standard fine-tuning in {} root scenario:'.format(num_roots))
            
            elif 'transformer_zero' in mode:
                print('RoBERTa-base zero shot in {} root scenario:'.format(num_roots))
            
            elif 'setfit_zero' in mode:
                print('SetFit zero shot in {} root scenario:'.format(num_roots))
             
            elif 'use_ft_transformer' in mode:
                print('RoBERTa-base two-step fine-tuning in {} root scenario:'.format(num_roots))
            
            elif 'setfit_pretrain' in mode:
                print('SetFit two-step fine-tuning in {} root scenario:'.format(num_roots))
                
            print(f1)

print("Job's done!")
           
