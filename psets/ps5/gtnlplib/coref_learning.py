import numpy as np
from collections import defaultdict
import coref

# deliverable 3.2
def mention_rank(markables,i,feats,weights):
    """ return top scoring antecedent for markable i

    :param markables: list of markables
    :param i: index of current markable to resolve
    :param feats: feature function
    :param weights: weight defaultdict
    :returns: index of best scoring candidate (can be i)
    :rtype: int

    """
    ## hide
    best_score = 0
    best_ant_idx = 0

    for idx in range(i+1):
        feat = feats(markables, i, idx)
        temp_score = 0
        for f,s in feat.items():
            temp_score += weights[f]*s
        if temp_score > best_score:
            best_score = temp_score
            best_ant_idx = idx
    return best_ant_idx
    
# deliverable 3.3
def compute_instance_update(markables,i,true_antecedent,feats,weights):
    """Compute a perceptron update for markable i.
    This function should call mention_rank to determine the predicted antecedent,
    and should make an update if the true antecedent and predicted antecedent *refer to different entities*

    Note that if the true and predicted antecedents refer to the same entity, you should not
    make an update, even if they are different.

    :param markables: list of markables
    :param i: current markable
    :param true_antecedent: ground truth antecedent
    :param feats: feature function
    :param weights: defaultdict of weights
    :returns: dict of updates
    :rtype: dict

    """
    # keep
    pred_antecedent = mention_rank(markables,i,feats,weights)

    ## possibly useful
    #print i,true_antecedent,pred_antecedent
    # print "i string: ", markables[i]
    # print "true: ", markables[true_antecedent], feats(markables,true_antecedent,i)
    # print "pred: ", markables[pred_antecedent], feats(markables,pred_antecedent,i)
    #print ""
    entity_match = markables[pred_antecedent]['entity'] == markables[true_antecedent]['entity']
    same_index = pred_antecedent == i and not true_antecedent == pred_antecedent
    if (same_index or not entity_match):

        update = defaultdict(float)
        truefeats = feats(markables,true_antecedent,i)
        predfeats = feats(markables,pred_antecedent,i)
        for f,s in truefeats.items():
            update[f] += s
        for f,s in predfeats.items():
            update[f] -= s
        return update
    else:
        return dict()


# deliverable 3.4
def train_avg_perceptron(markables,features,N_its=20):
    # the data and features are small enough that you can
    # probably get away with naive feature averaging

    weights = defaultdict(float)
    tot_weights = defaultdict(float)
    weight_hist = []
    T = 0.

    for it in xrange(N_its):
        num_wrong = 0 #helpful but not required to keep and print a running total of errors
        for document in markables:
            true_ants = coref.get_true_antecedents(document)
            for i in range(len(document)):
                updates = compute_instance_update(document, i,true_ants[i], features, weights)
                if not len(updates) == 0:
                    print "updates at: ", i
                    print updates
                    num_wrong += 1
                for k,v in updates.items():
                    weights[k] += v
                    tot_weights[k] += T*v
                T += 1.0
        print num_wrong,

        # update the weight history
        weight_hist.append(defaultdict(float))
        for feature in tot_weights.keys():
            weight_hist[it][feature] = tot_weights[feature]/T

    return weight_hist

# helpers
def make_resolver(features,weights):
    return lambda markables : [mention_rank(markables,i,features,weights) for i in range(len(markables))]
        
def eval_weight_hist(markables,weight_history,features):
    scores = []
    for weights in weight_history:
        score = coref.eval_on_dataset(make_resolver(features,weights),markables)
        scores.append(score)
    return scores
