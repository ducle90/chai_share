import fileinput
import numpy as np

def main():
    # Receive a stream of numbers through stdin, output stats
    nums = [float(x) for x in fileinput.input()]
    print 'Count\tSum'
    print '{}\t{}'.format(len(nums), np.sum(nums))
    print
    print 'Mean\tMedian\tStdev'
    print '{}\t{}\t{}'.format(np.mean(nums), np.median(nums), np.std(nums))
    print
    print '25th\t75th\t25th-75th Range'
    print '{}\t{}\t{}'.format(
        np.percentile(nums, 25), np.percentile(nums, 75),
        np.percentile(nums, 75) - np.percentile(nums, 25)
    )
    print
    print 'Min\tMax\tMin-Max Range'
    print '{}\t{}\t{}'.format(
        np.min(nums), np.max(nums), np.max(nums) - np.min(nums)
    )
    print

if __name__ == '__main__':
    main()
