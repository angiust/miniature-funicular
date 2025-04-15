# miniature-funicular

simulation:

for t in 0.{0..9} 1.0; do ./simulate.py -T "$t" -t 100 > output/output_"$t".csv & done

for t in 0 0.1 0.01; do for a in 0.{1..9} 1.0; do ./simulate.py -a "$a" -T "$t" -t 100 > output/output_a"$a"_T"$t".csv & done; done

for t in 0.{0..9} 1.0; do ./simulate.py -T "$t" -t 100 --mix > output_mix/output_mix_"$t".csv & done

for t in 0 0.1 0.01; do for a in 0.{1..9} 1.0; do ./simulate.py -a "$a" -T "$t" -t 100 --mix > output_mix/output_mix_a"$a"_T"$t".csv & done; done

save plot:

for f in output/*.csv; do base=$(basename "$f" .csv); ./plot.py --title "$base" --mode savefig --output "plot/${base}.png" < "$f" & done

parallel_version: ls output/*.csv | parallel --jobs 4 'base=$(basename {} .csv); ./plot.py --title "$base" --mode savefig --output plot/"$base".png < {}'

ls output_mix/*.csv | parallel --jobs 4 'base=$(basename {} .csv); ./plot.py --title "$base" --mode savefig --output plot_mix/"$base".png < {}'
