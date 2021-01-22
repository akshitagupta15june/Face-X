import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ds = [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2304]

with open('outputs/results.txt') as f:
    lines = f.readlines()
    accuracies = [list(map(float, line.split(',')[2].split(' ')))
                  for line in lines]
    ols_train_accuracies = accuracies[0]
    ols_test_accuracies = accuracies[1]
    ridge_train_accuracies = accuracies[2]
    ridge_test_accuracies = accuracies[3]

print(' | '.join(['%.02f' % acc for acc in ols_train_accuracies]))
print(' | '.join(['%.02f' % acc for acc in ols_test_accuracies]))
print(' | '.join(['%.02f' % acc for acc in ridge_train_accuracies]))
print(' | '.join(['%.02f' % acc for acc in ridge_test_accuracies]))

plt.title('Performance of Featurized Ordinary Least Squares')
plt.plot(ds, ols_train_accuracies, label='train')
plt.plot(ds, ols_test_accuracies, label='test')
plt.legend()
plt.savefig('outputs/perf_feat_ols.png')

plt.figure()
plt.title('Performance of Featurized Ridge, Ordinary Least Squares')
plt.plot(ds, ols_train_accuracies, label='ols train')
plt.plot(ds, ols_test_accuracies, label='ols test')
plt.plot(ds, ridge_train_accuracies, label='ridge train')
plt.plot(ds, ridge_test_accuracies, label='ridge test')
plt.legend()
plt.savefig('outputs/perf_feat_ridge_ols.png')