import statistics
from sklearn.metrics import accuracy_score, f1_score


def evaluate_yes_no(predicted, target):
    accuracy = accuracy_score(y_true=target, y_pred=predicted)

    f1_y = f1_score(y_true=target, y_pred=predicted,
                    average='binary', pos_label=1)
    f1_n = f1_score(y_true=target, y_pred=predicted,
                    average='binary', pos_label=0)
    macro_average_f_measure = (f1_y+f1_n)/2

    return {'accuracy': accuracy, 'macro_average_f_measure': macro_average_f_measure}


def evaluate_factoid(predicted, target):
    c_1 = 0
    c_5 = 0
    rank_sum = 0
    for i, elem in enumerate(target):
        # Accuracy
        if predicted[i][0] == target[i]:
            c_1 += 1
        if target[i] in predicted[i]:
            c_5 += 1

        # Mean reciprocal rank
        try:
            pos = predicted[i].index(target[i])
            rank_sum += 1/(pos+1)
        except ValueError:
            pass

    return {"strict_accuracy": c_1/len(target), "lenient_accuracy": c_5/len(target), "mean_reciprocal_rank": rank_sum/len(target)}


def evaluate_list(predicted, target):
    # TODO: Gestione dei sinonimi

    # no duplicates
    # TP in both list
    #FP in pred not in target
    #FN not in pred in target
    precision_list = []
    recall_list = []
    f1_list = []
    for i, sample in enumerate(predicted):
        tp = 0
        fp = 0
        fn = 0
        local_pred = set(sample)
        local_tar = set(target[i])
        union = local_pred.union(local_tar)
        for word in union:
            if(word in local_pred and word in local_tar):
                tp += 1
            else:
                if(word in local_pred and word not in local_tar):
                    fp += 1
                else:
                    fn += 1
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(2*((precision*recall)/(precision+recall)))

    return {"mean_precison": statistics.mean(precision_list), "mean_recall": statistics.mean(recall_list), "mean f1": statistics.mean(f1_list)}
