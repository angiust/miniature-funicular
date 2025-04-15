# miniature-funicular

for t in 0.{1..9} 1.0; do ./simulate.py -T "$t" -t 10 > output/output_"$t".csv ; done
