import csv
from numpy import mean, std

header = ['max.ncats', 'runnum', 'iterations', 'ncats', 'likelihood', 'mean.class.size','sd.class.size']
fnameout = 'brown_tests_uniform.csv'

with open(fnameout,'w') as fout:
    fwriter = csv.writer(fout)
    fwriter.writerow(header)

    with open('nb_tests.o725059') as f:
        freader = csv.reader(f)
        for row in freader:
            fields = row[:-2]
            class_counts = eval(row[-2])

            mean_size = mean(class_counts)
            sd_size = std(class_counts)

            rowout = fields + [mean_size, sd_size]
            fwriter.writerow(rowout)
