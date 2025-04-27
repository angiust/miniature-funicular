# miniature-funicular

## simulation:

### every pattern:

data:

parallel --bar --jobs 2 './run/one_sample_every_pattern_init.py -T {1} -a {2} > outputs/data/everyPattern/everyPatternInit_T{1}_a{2}_1.csv' ::: 0 0.1 0.01 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

plot:

parallel --bar --jobs 4 'base=$(basename {} .csv); ./plot/every_pattern_plot.py --title "{=s/.csv//=}" --output outputs/plot/everyPattern/"$base".png < {}' ::: outputs/data/everyPattern/*.csv

### first pattern:

data:
parallel --bar --jobs 2 './run/one_sample_every_pattern_init.py -T {1} -a {2} > outputs/data/firstPattern/firstPattern_T{1}_a{2}_1.csv' ::: 0 0.1 0.01 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

plot:
parallel --bar --jobs 4 'base=$(basename {} .csv); ./plot/plot.py --title "{=s/.csv//=}" --output outputs/plot/firstPattern/"$base".png < {}' ::: outputs/data/firstPattern/*.csv

### mixture:

data:
parallel --bar --jobs 2 './run/simulate.py --init_type mixture -T {1} -a {2} > outputs/data/mixtureInit/mixtureInit_T{1}_a{2}_1.csv' ::: 0 0.1 0.01 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

plot:
parallel --bar --jobs 4 'base=$(basename {} .csv); ./plot/plot.py --title "{=s/.csv//=}" --output outputs/plot/mixtureInit/"$base".png < {}' ::: outputs/data/mixtureInit/*.csv

### magnetization hystogram:

data:
parallel --bar --jobs 2 './run/hystograms.py -s 100 -T {1} -a {2} > outputs/data/magnHyst/magnHyst_T{1}_a{2}_1.csv' ::: 0 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

plot:
parallel --bar --jobs 4 'base=$(basename {} .csv); ./plot/hystogram_plot.py --title "{=s/.csv//=}" --output outputs/plot/magnHyst/"$base".png < {}' ::: outputs/data/magnHyst/*.csv

### random init:

data:
parallel --bar --jobs 2 './run/simulate.py --init_type random -T {1} -a {2} > outputs/data/randomInit/randomInit_T{1}_a{2}_1.csv' ::: 0 0.1 0.01 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

plot:
parallel --bar --jobs 4 'base=$(basename {} .csv); ./plot/plot.py --title "{=s/.csv//=}" --output outputs/plot/randomInit/"$base".png < {}' ::: outputs/data/randomInit/*.csv

### vary sample not averaged random init:



### first pattern old:

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
