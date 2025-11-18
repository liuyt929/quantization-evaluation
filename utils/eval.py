import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM

def evaluation(model,tokenizer):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    # task_names = lm_eval_utils.pattern_match(["winogrande","arc_easy","arc_challenge"], ALL_TASKS)

    results = lm_eval.simple_evaluate(hflm, tasks=[
        # "gsm8k",
        "mmlu",
        "mathqa",
        "arc_challenge","hellaswag","winogrande","boolq","lambada_openai","piqa","sst2"
            ],batch_size=1)['results']
    print("finish")
    print(results)
    with open('result.txt', 'w') as f:
        f.write(str(results))
    # metric_vals = {}
    # for task, result in results:
    #     if 'acc_norm,none' in result:
    #         metric_vals[task] = round(result['acc_norm,none'], 4)
    #     elif 'acc,none' in result:
    #         metric_vals[task] = round(result['acc,none'], 4)
    #     else:
    #         print(f"[警告] {task} 没有 acc_norm,none 或 acc,none")

    # 计算平均值
    # if metric_vals:
    #     metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals), 4)
    # print(metric_vals)
    # metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    
    # metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    # print(metric_vals)
