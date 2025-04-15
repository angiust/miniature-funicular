# miniature-funicular

for t in 0.{0..9} 1.0; do ./simulate.py -T "$t" -t 100 > output/output_"$t".csv & done

for t in 0 0.1 0.01; do for a in 0.{1..9} 1.0; do ./simulate.py -a "$a" -T "$t" -t 100 > output/output_a"$a"_T"$t".csv & done; done
