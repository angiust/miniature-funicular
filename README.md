# miniature-funicular

## simulation:

### first pattern:

for t in 0.{0..9} 1.0; do ./simulate.py -T "$t" -t 100 > output/output_"$t".csv & done

for t in 0 0.1 0.01; do for a in 0.{1..9} 1.0; do ./simulate.py -a "$a" -T "$t" -t 100 > output/output_a"$a"_T"$t".csv & done; done

parallel --jobs 4 './simulate.py -a {1} -T {2} -t 100 > one_pattern/output_a{1}_T{2}.csv' ::: 0.{1..9} 1.0 ::: 0 0.1 0.01 &

    (time parallel --bar --jobs 4 './simulate.py -a {1} -T {2} -t 100 -s 1 > one_sample_one_pattern/output_a{1}_T{2}.csv' ::: 0.{1..9} 1.0 ::: 0 0.1 0.01) &

(time ls one_pattern/*.csv | parallel --bar --jobs 4 'base=$(basename {} .csv); ./plot.py --title "{=s/.csv//=}" --output one_pattern_plot/"$base".png < {}') &

### binary mixture:

for t in 0.{0..9} 1.0; do ./simulate.py -T "$t" -t 100 --mix > output_mix/output_mix_"$t".csv & done

for t in 0 0.1 0.01; do for a in 0.{1..9} 1.0; do ./simulate.py -a "$a" -T "$t" -t 100 --mix > output_mix/output_mix_a"$a"_T"$t".csv & done; done

### for continuous mixture:

for t in 0 0.1 0.01; do for a in 0.{1..9} 1.0; do ./simulate.py -a "$a" -T "$t" -t 100 --mix > output_c_mix/output_c_mix_a"$a"_T"$t".csv & done; done

#### parallel version:

parallel --will-cite -j 0 'a={1}; t={2}; ./simulate.py -a "$a" -T "$t" -t 100 --mix > output_c_mix/output_c_mix_a${a}_T${t}.csv' ::: 0.{1..9} 1.0 ::: 0 0.1 0.01 &


# save plot:

for f in output/*.csv; do base=$(basename "$f" .csv); ./plot.py --title "$base" --mode savefig --output "plot/${base}.png" < "$f" & done

parallel_version: ls output/*.csv | parallel --jobs 4 'base=$(basename {} .csv); ./plot.py --title "$base" --mode savefig --output plot/"$base".png < {}'

(time ls one_pattern/*.csv | parallel --bar --jobs 4 'base=$(basename {} .csv); ./plot.py --title "{=s/.csv//=}" --output one_pattern_plot/"$base".png < {}') &

ls output_mix/*.csv | parallel --jobs 4 'base=$(basename {} .csv); ./plot.py --title "$base" --mode savefig --output plot_mix/"$base".png < {}'

ls output_c_mix/*.csv | parallel --jobs 4 'base=$(basename {} .csv); ./plot.py --title "$base" --mode savefig --output plot_c_mix/"$base".png < {}' &
