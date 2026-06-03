# ad_scoring.py
# Predicting ad creative (headline) performance using token-based scoring
# Adapted from Vincent Granville's nlp_scoring.py (Section 8.3.3)
# Inspired by GenAITechLab Taxonomy LLM project
#
# USE CASE: DSP ad creative optimization (display, native, programmatic)
# Uses Microsoft MIND news dataset (real headlines + real click data)
# as a proxy for ad headlines — structurally identical problem.
#
# Dataset: 14,940 news articles with impression-weighted performance scores
# from the MIND (Microsoft News) recommendation dataset.

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

mpl.rcParams['lines.linewidth'] = 0.3
mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7


# --- [1] Read ad performance data

filename = "Ad-Performance.txt"
data = pd.read_csv(filename, sep='\t', engine='python', encoding='utf-8')

arr_headlines = data['Title']       # ad headline / creative copy
arr_vertical = data['Status']       # ad vertical / subcategory
arr_perf = data['Page views']       # performance metric (impressions * CTR boost)
arr_url = data['URL']               # ad landing page
arr_publisher = data['Author']      # publisher / subcategory
arr_categories = {}

category_perf = {}
category_count = {}
hash_words_count = {}
hash_perf = {}
hash_headlines = {}
hash_publisher_count = {}


# --- [2] Core functions

def compress_vertical(vertical):
    vertical = str(vertical).lower()
    if vertical in ('sports', 'football', 'basketball', 'baseball', 'soccer', 'tennis',
                     'golf', 'hockey', 'motorsport', 'boxing_mma', 'cricket'):
        return 'sports'
    elif vertical in ('news', 'newspolitics', 'newsworld', 'newsus', 'newscrime',
                       'newsopinion', 'newsscience'):
        return 'news'
    elif vertical in ('finance', 'financecompanies', 'financemarkets', 'financepersonal'):
        return 'finance'
    elif vertical in ('health', 'medical', 'weightloss', 'nutrition', 'voices'):
        return 'health'
    elif vertical in ('lifestyle', 'lifestyleroyals', 'lifestylebuzz'):
        return 'lifestyle'
    elif vertical in ('entertainment', 'movies', 'music', 'tv', 'celebrity'):
        return 'entertainment'
    elif vertical in ('autos', 'autosenthusiasts', 'autosclassics'):
        return 'autos'
    elif vertical in ('foodanddrink', 'recipes', 'wine', 'restaurants'):
        return 'food'
    elif vertical in ('travel', 'traveltripideas', 'travelnews'):
        return 'travel'
    return 'other'

def compress_channel(url):
    url = str(url).lower()
    if 'msn.com' in url:
        return 'msn'
    return 'partner'

def update_hash(headlineID, word, hash_words_count, hash_perf, hash_headlines):
    if word in hash_words_count:
        hash_words_count[word] += 1
        hash_perf[word] += perf
        hash_headlines[word][headlineID] = perf
    else:
        hash_words_count[word] = 1
        hash_perf[word] = perf
        hash_headlines[word] = {headlineID: perf}
    return hash_words_count, hash_perf, hash_headlines

def extract_single_tokens(hID, words, hwc, hp, hh):
    for word in words:
        hwc, hp, hh = update_hash(hID, word, hwc, hp, hh)
    return hwc, hp, hh

def extract_adjacent_phrases(hID, words, hwc, hp, hh):
    for idx in range(len(words)):
        w = words[idx]
        if idx < len(words)-1 and words[idx+1] != '':
            bigram = w + "~" + words[idx+1]
            hwc, hp, hh = update_hash(hID, bigram, hwc, hp, hh)
            if idx < len(words)-2 and words[idx+2] != '':
                trigram = bigram + "~" + words[idx+2]
                hwc, hp, hh = update_hash(hID, trigram, hwc, hp, hh)
    return hwc, hp, hh

def extract_cooccurring_tokens(hID, words, hwc, hp, hh):
    param_D = 1
    for k in range(len(words)):
        for l in range(len(words)):
            w1, w2 = words[k], words[l]
            if w1 < w2 and abs(k-l) > param_D and w1 != '' and w2 != '':
                hwc, hp, hh = update_hash(hID, w1 + "^" + w2, hwc, hp, hh)
    return hwc, hp, hh

def get_normalized_perf(hID, arr_perf):
    return np.log(max(float(arr_perf[hID]), 1.0))


# --- [3] Time-adjust performance (not needed for MIND — single time window)
# Keeping the structure for when you plug in real DSP data with date ranges

param_T1 = 0.00  # set to 0 = no time adjustment (MIND data is single snapshot)
param_T2 = 0.11
arr_perf_adj = np.zeros(len(arr_perf))
for k in range(len(arr_perf)):
    boost = param_T1 * np.sqrt(k + param_T2 * len(arr_perf))
    arr_perf_adj[k] = arr_perf[k] * (1 + boost)
arr_perf = np.copy(arr_perf_adj)


# --- [4] Build token tables from ad headlines

for k in range(len(data)):
    pub = str(arr_publisher[k])
    hash_publisher_count[pub] = hash_publisher_count.get(pub, 0) + 1

param_A = 100  # publishers with fewer entries are bundled

for k in range(len(data)):
    perf = get_normalized_perf(k, arr_perf)
    vertical = compress_vertical(arr_vertical[k])
    channel = compress_channel(arr_url[k])
    category = channel + "~" + vertical
    pub = str(arr_publisher[k])
    if hash_publisher_count[pub] > param_A:
        arr_categories[k] = category + "~" + pub
    else:
        arr_categories[k] = category

    words = str(arr_headlines[k]).replace(',', ' ').replace(':', ' ').replace('?', ' ')
    words = words.replace('.', ' ').replace('(', ' ').replace(')', ' ')
    words = words.replace('-', ' ').replace("'s", '').replace("'", '')
    words = words.replace('  ', ' ').replace('\xa0', ' ')
    words = words.split(' ')

    hash_words_count, hash_perf, hash_headlines = extract_single_tokens(
        k, words, hash_words_count, hash_perf, hash_headlines)
    hash_words_count, hash_perf, hash_headlines = extract_adjacent_phrases(
        k, words, hash_words_count, hash_perf, hash_headlines)
    hash_words_count, hash_perf, hash_headlines = extract_cooccurring_tokens(
        k, words, hash_words_count, hash_perf, hash_headlines)

mean_perf = sum(hash_perf.values()) / sum(hash_words_count.values())
print("Mean normalized performance: %6.3f" % mean_perf)
print("Total headlines: %d" % len(data))
print("Unique tokens extracted: %d" % len(hash_words_count))


# --- [5] Rank tokens by performance

eps = 1e-12
hash_perf_rel = {}
for word in hash_perf:
    hash_perf_rel[word] = hash_perf[word] / hash_words_count[word]
hash_perf_rel = dict(sorted(hash_perf_rel.items(), key=lambda item: item[1], reverse=True))

hash_perf_deduped = {}
old_perf = -1
old_word = ''
for word in hash_perf_rel:
    p = hash_perf_rel[word]
    if abs(p - old_perf) > eps:
        if old_perf != -1:
            hash_perf_deduped[old_word] = old_perf
        old_word = word
        old_perf = p
    else:
        if len(word) > len(old_word):
            old_word = word
hash_perf_deduped[old_word] = old_perf

print("\n=== HIGH-PERFORMANCE AD HEADLINE TOKENS")
print("   (tokens correlated with high/low click performance)\n")
print(" Count  Rel.Perf  Token")
print(" -----  --------  -----")
top_count = 0
for word in hash_perf_deduped:
    count = hash_words_count[word]
    if count > 30 or (count > 8 and '~' in word):
        rel = hash_perf_rel[word] / mean_perf
        marker = "  << HIGH" if rel > 1.10 else ("  << LOW" if rel < 0.90 else "")
        print("%6d   %5.3f   %s%s" % (count, rel, word, marker))
        top_count += 1
        if top_count >= 60:
            print("  ... (showing top 60)")
            break


# --- [6] Average performance per category (vertical + channel)

for k in range(len(data)):
    category = arr_categories[k]
    p = get_normalized_perf(k, arr_perf)
    if category in category_count:
        category_perf[category] += p
        category_count[category] += 1
    else:
        category_perf[category] = p
        category_count[category] = 1

print("\n\n=== VERTICAL PERFORMANCE\n")
for category in sorted(category_count, key=lambda c: category_count[c], reverse=True):
    count = category_count[category]
    category_perf[category] /= count
    print("%5d %6.3f %s" % (count, category_perf[category], category))


# --- [7] Short list for clustering

short_list = {}
param_G1 = 1.08
param_C1 = 15
param_C2 = 6

for word in hash_perf_deduped:
    count = hash_words_count[word]
    avg_perf = hash_perf[word] / count
    if avg_perf > param_G1 * mean_perf:
        if count > param_C1 or (count > param_C2 and '~' in word):
            short_list[word] = 1

print("\n%d high-performing tokens selected for clustering" % len(short_list))


# --- [8] Token similarity

hash_pairs = {}
aux_list = {}
param_S = 0.15

for word1 in short_list:
    set1 = set(hash_headlines[word1].keys())
    for word2 in short_list:
        if word1 >= word2:
            continue
        set2 = set(hash_headlines[word2].keys())
        sim = len(set1 & set2) / len(set1 | set2)
        if sim > param_S:
            hash_pairs[(word1, word2)] = sim
            hash_pairs[(word2, word1)] = sim
            hash_pairs[(word1, word1)] = 1.0
            hash_pairs[(word2, word2)] = 1.0
            aux_list[word1] = 1
            aux_list[word2] = 1


# --- [9] Cluster into creative themes

param_N = 20
n = len(aux_list)
arr_word = list(aux_list.keys())

if n >= param_N:
    dist_matrix = [[1 - hash_pairs.get((arr_word[i], arr_word[j]), 0)
                     for j in range(n)] for i in range(n)]

    from sklearn.cluster import AgglomerativeClustering
    hierarch = AgglomerativeClustering(n_clusters=min(param_N, n), linkage='average').fit(dist_matrix)

    groups = hierarch.labels_
    hash_group_words = {}
    for k in range(len(groups)):
        g = groups[k]
        w = arr_word[k]
        if g in hash_group_words:
            hash_group_words[g] = (*hash_group_words[g], w)
        else:
            hash_group_words[g] = (w,)

    print("\n\n=== AD CREATIVE THEMES (what headline patterns drive clicks)\n")
    for g in sorted(hash_group_words):
        tokens = hash_group_words[g]
        all_headlines = set()
        for t in tokens:
            all_headlines.update(hash_headlines[t].keys())
        perfs = [get_normalized_perf(h, arr_perf) for h in all_headlines]
        avg = np.mean(perfs) if perfs else 0
        print("Theme %2d (avg perf=%.2f, %d headlines):" % (g, avg, len(all_headlines)))
        print("  Tokens: %s" % str(tokens[:8]))
        shown = 0
        for hid in sorted(all_headlines, key=lambda h: get_normalized_perf(h, arr_perf), reverse=True):
            if shown >= 5:
                break
            print("    %6.2f  %s" % (get_normalized_perf(hid, arr_perf), str(arr_headlines[hid])[:80]))
            shown += 1
        print()

    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(dist_matrix)
    plt.figure(figsize=(14, 6))
    dendrogram(Z)
    plt.title("Ad Headline Token Clusters (Creative Themes)")
    plt.savefig("ad_creative_themes.png", dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("\nToo few tokens for clustering (%d), skipping." % n)


# --- [10] Predict performance for each headline

reversed_hash = {}
for word in hash_headlines:
    perf_rel = hash_perf_rel[word]
    for hid in hash_headlines[word]:
        if hid not in reversed_hash:
            reversed_hash[hid] = {}
        reversed_hash[hid][word] = perf_rel

observed = []
predicted = []
missed = 0
count_n = 0
param_W1 = 1
param_W2 = 2.00

for hid in reversed_hash:
    p = get_normalized_perf(hid, arr_perf)
    rhash = reversed_hash[hid]
    count_n += 1
    wt_sum = 0
    wt_count = 0
    for word in rhash:
        weight = hash_words_count[word]
        if weight > param_W1:
            w = (1/weight) ** param_W2
            wt_count += w
            wt_sum += w * rhash[word]
    if wt_count > 0:
        est = wt_sum / wt_count
    else:
        missed += 1
        est = category_perf.get(arr_categories.get(hid, ''), mean_perf)
    observed.append(p)
    predicted.append(est)

observed = np.array(observed)
predicted = np.array(predicted)
mean_obs = np.mean(observed)

min_loss = 1e9
best_Z = 0
for test_Z in np.arange(-0.50, 0.50, 0.05):
    scaled = predicted + test_Z * (predicted - mean_obs)
    loss = max(abs(np.quantile(observed, q) - np.quantile(scaled, q)) for q in (.10, .25, .50, .75, .90))
    if loss < min_loss:
        min_loss = loss
        best_Z = test_Z

predicted = predicted + best_Z * (predicted - mean_obs)
mae = np.mean(np.abs(observed - predicted))
correl = np.corrcoef(observed, predicted)[0][1]

plt.figure()
plt.axline((min(observed), min(observed)), (max(observed), max(observed)), c='red')
plt.scatter(predicted, observed, s=0.3, c="steelblue", alpha=0.3)
plt.title("Ad Headlines: Observed vs Predicted Performance (%d ads)" % count_n)
plt.xlabel("Predicted performance")
plt.ylabel("Observed performance")
plt.savefig("ad_observed_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n=== AD HEADLINE PERFORMANCE PREDICTIONS\n")
print("Predicted vs observed for %d ad headlines\n" % count_n)
print("Loss (KS approx):   %6.3f" % min_loss)
print("Missed headlines:    %d out of %d" % (missed, count_n))
print("Mean perf (obs):    %8.3f" % mean_obs)
print("Mean perf (pred):   %8.3f" % np.mean(predicted))
print("Mean absolute error:%8.3f" % mae)
print("Correlation:        %8.3f" % correl)
print()
print("Quantile comparison (observed | predicted):")
for q in (.10, .25, .50, .75, .90):
    print("  P.%02d: %8.3f  %8.3f" % (int(q*100), np.quantile(observed, q), np.quantile(predicted, q)))

print("\nPlots saved: ad_creative_themes.png, ad_observed_vs_predicted.png")
print("\n--- HOW TO USE WITH REAL DSP DATA ---")
print("1. Export ad headlines + impressions + clicks from your DSP")
print("2. Compute performance = impressions * (1 + 10*CTR) or use revenue")
print("3. Format as tab-separated: Headline \\t URL \\t Publisher \\t Performance \\t Date \\t Vertical")
print("4. Replace 'Ad-Performance.txt' with your file")
print("5. Adjust compress_vertical() for your verticals")
print("6. Re-run this script")
