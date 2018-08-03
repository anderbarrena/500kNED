use warnings;
use strict;
use Getopt::Long qw(:config auto_help);
use File::Basename;
use List::Util 'shuffle';

binmode STDOUT, ":utf8";
my ($COLLECT,$TEST,$W2V,$DICT,$CLUSTER,$TRAIN_M0THS,$PRIOR,$PATH,$SUBSET,$WINDOW,
$TAG,$NN,$CASE,$GENERATIVE,$OUT,$TO_PEC,$PES,$DOC,$AUX_DICT);

GetOptions("COLLECT=s" => \$COLLECT, "TEST=s" => \$TEST, "W2V=s" => \$W2V, "DICT=s" => \$DICT,
"CLUSTER" => \$CLUSTER, "TRAIN_M0THS" => \$TRAIN_M0THS, "PRIOR" => \$PRIOR, "PATH=s" => \$PATH, 
"SUBSET=s" => \$SUBSET, "WINDOW=f" => \$WINDOW, "TAG=s" => \$TAG, "NN=s" => \$NN, "CASE" => \$CASE, 
"OUT=s" => \$OUT, "GENERATIVE" => \$GENERATIVE, "TO_PEC" => \$TO_PEC, "DOC" => \$DOC, "PES" => \$PES,
"AUX_DICT=s" => \$AUX_DICT);

my %w2v = ();
my %kode = ();
my %prior = ();
my %aux_dict = ();
my %entities = ();
my $path = "";
my $window = 50;
my $instance_size = 5000;
my $M = 151441649;
my $tag = "";
my $nn = "\%lstm.py";

&init();
if($OUT)
{
    &doG($OUT) if $GENERATIVE;
    if($TO_PEC)
    {
	&loadS($DICT,\%prior,\%entities,\%kode);
	&do2pec($OUT);
    }
    if($PES)
    {
	&loadS($DICT,\%prior,\%entities,\%kode);
	&pes($OUT);
    }
} 
if($SUBSET)
{
    &subset($W2V,$SUBSET);
}
if($COLLECT)
{
    &loadS($DICT,\%prior,\%entities,\%kode);
    &collect($COLLECT);
}
if($CLUSTER)
{
    &loadS($DICT,\%prior,\%entities,\%kode);
    &loadV($W2V,\%w2v);
    &cluster();
}
if($TRAIN_M0THS)
{
    &loadS($DICT,\%prior,\%entities,\%kode) unless $CLUSTER;
    &trainM0ths();
}
if($TEST)
{
    &loadS($DICT,\%prior,\%entities,\%kode);
    &loadSA($AUX_DICT,\%aux_dict);
    &loadV($W2V,\%w2v);
    &loadE($TEST);
    &testM0ths($TEST);
}
sub init
{
    $window = $WINDOW if $WINDOW;
    $nn = $NN if defined($NN);
    $tag = ".".$TAG if defined($TAG);
    $tag .= ".w".$window;
    my $v = basename($W2V) if $W2V;
    $v =~ s/\.bz2//g;
    $tag .= ".".$v if $W2V;
    $path = "$PATH" if $PATH;
    mkdir $path."ssv/";
    mkdir $path."ssv/train/" if $COLLECT || $CLUSTER;
    mkdir $path."ssv/train/entities/" if $COLLECT;
    mkdir $path."m\@ths/" if $TRAIN_M0THS;
    mkdir $path."ssv/test/" if $TEST;
    if (!$OUT)
    {
	warn "\n
    \t                                       88           
    \t                                 ,d    88           
    \t                                 88    88           
    \t 88,dPYba,,adPYba,   ,adPPYba, MM88MMM 88,dPPYba,   
    \t 88P\'   \"88\"    \"8a a8\"     \"8a  88    88P\'    \"8a  
    \t 88      88      88 8b   0   d8  88    88       88  
    \t 88      88      88 \"8a,   ,a8\"  88,   88       88  
    \t 88      88      88  \`\"YbbdP\"\'   \"Y888 88       88 Nst3r 
    \t
    \t
    \nDeep Neural Net based One4all Entity Linking model at --> ".localtime()."\n";
	
	warn "\n\tFeatures:\n";
	warn "\t\t-> vector subset for ".$SUBSET."\n" if $SUBSET;
	warn "\t\t-> cased context\n" if $CASE;
	warn "\t\t-> collecting entities\n" if $COLLECT;
	warn "\t\t-> clustering entities\n" if $CLUSTER;
	warn "\t\t-> generative model\n" if $GENERATIVE;
	warn "\t\t-> ".$nn." m\@ths\n" if $NN;
	warn "\t\t-> testing m\@ths\n" if $TEST;
	warn "\t\t-> using Doc mentions as context\n" if $TEST && $DOC;
	warn "\t\t-> combining with prior\n" if $PRIOR && $TEST;
	warn "\n\tNow:\n";
    }
}

sub doG
{
    my ($out) = @_;
    my %print = ();
    open(Fitx,"$out") || die;
    binmode Fitx, ":utf8";
    while (<Fitx>)
    {
	chomp;
	my($doc,$id,$name,$ents,$flag) = split("\t",$_);
	foreach my $e_c (split(" ",$ents))
	{
	    $e_c =~ s/::::/:::/g;
	    my($e,$c) = split(":::",$e_c);
	    $entities{$e} = 1;
	    $print{$doc."\t".$id."\t".$name}{$e}=$c;
	}
    }
    close(Fitx);
    my %e_c = ();
    # warn "\t\t-> reading priors\n";
    foreach my $ent (keys %entities)
    {
	my ($ssv) = &ePaths($ent);
	my $l = 0;
	my $p = 0;
	if(-e $ssv.".bz2")
	{
	    open (I, "-|:encoding(UTF-8)", "bzcat ".$ssv.".bz2") || die $! ;
	    while (<I>) 
	    {	
		chomp;
		$l++;    
	    }
	    close(I);
	}
	$e_c{$ent} = $l + 1;
	$entities{$ent} /= $e_c{$ent};
    }
    foreach my $head (keys %print)
    {
	foreach my $e (keys %{$print{$head}})
	{
	    $print{$head}{$e} += log($entities{$e});  
	}
    }
    # warn "\t\t-> modeling output\n\n";
    foreach my $head (keys %print)
    {
	print $head."\t";
	foreach my $e (sort {$print{$head}{$b} <=> $print{$head}{$a}} keys %{$print{$head}})
	{
	    next if $e eq "NIL";
	    print $e.":::".$print{$head}{$e}." ";
	}
	print "NIL:::".$print{$head}{"NIL"}."\n";
    }
}

sub do2pec
{
    my ($out) = @_;
    my %print = ();
    open(Fitx,"$out") || die;
    binmode Fitx, ":utf8";
    while (<Fitx>)
    {
	chomp;
	my($doc,$id,$name,$ents,$flag) = split("\t",$_);
	foreach my $e_c (split(" ",$ents))
	{
	    $e_c =~ s/::::/:::/g;
	    my($e,$c) = split(":::",$e_c);
	    $print{$doc."\t".$id."\t".$name}{$e}=$c-log($prior{$name}{$e}) if $e ne "NIL";
	    $print{$doc."\t".$id."\t".$name}{$e}=$c-log(0.5) if $e eq "NIL";
	}
    }
    close(Fitx);
    # warn "\t\t-> modeling output\n\n";
    foreach my $head (keys %print)
    {
	print $head."\t";
	foreach my $e (sort {$print{$head}{$b} <=> $print{$head}{$a}} keys %{$print{$head}})
	{
	    next if $e eq "NIL";
	    print $e.":::".$print{$head}{$e}." ";
	}
	print "NIL:::".$print{$head}{"NIL"}."\n";
    }
}

sub pes
{
    my ($out) = @_;
    my %print = ();
    open(Fitx,"$out") || die;
    binmode Fitx, ":utf8";
    while (<Fitx>)
    {
	chomp;
	my($doc,$id,$name,$ents,$flag) = split("\t",$_);
	foreach my $e_c (split(" ",$ents))
	{
	    $e_c =~ s/::::/:::/g;
	    my($e,$c) = split(":::",$e_c);
	    $print{$doc."\t".$id."\t".$name}{$e}=log($prior{$name}{$e}) if $e ne "NIL";
	    $print{$doc."\t".$id."\t".$name}{$e}=log(0.5) if $e eq "NIL";
	}
    }
    close(Fitx);
    # warn "\t\t-> modeling priors output\n\n";
    foreach my $head (keys %print)
    {
	print $head."\t";
	foreach my $e (sort {$print{$head}{$b} <=> $print{$head}{$a}} keys %{$print{$head}})
	{
	    next if $e eq "NIL";
	    print $e.":::".$print{$head}{$e}." ";
	}
	print "NIL:::".$print{$head}{"NIL"}."\n";
    }
}

sub subset
{
    my ($file,$subset) = @_;
    warn "\t\t-> creating subset for <".basename($subset)."> at --> ".localtime()."\n";
    my @lines = ();
    my %words;
    open (I, "-|:encoding(UTF-8)", "bzcat $subset") || die $! ;
    while (<I>) 
    {	
	chomp;
	push(@lines,$_);
    }
    close(I);
    foreach my $line (@lines)
    {
	my($docid,$id,$o,$m,$mm,$hm,$type,$w_ctx,$m_ctx,$m_fd) = split (/\t/,$line); 
	my $w = &getW($m,$mm,$hm);                                                
	my $ctxs;
	foreach my $tok (split("_",$w))
	{
	    $words{$tok}++;
	}
	($ctxs) = &getAnchor($w_ctx,$w); 
	foreach my $ctx (@$ctxs)
	{
	    foreach my $tok (split(" ",$ctx))
	    {
		$words{$tok}++;
	    }
	}
	($ctxs) = &getDoc($m_fd,$w);
	foreach my $ctx (@$ctxs)
	{
	    foreach my $tok (split(" ",$ctx))
	    {
		$words{$tok}++;
	    }
	}
    }    
    warn "\t\t-> loading word2vect <".basename($file)."> at --> ".localtime()."\n";
    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $! ;
    while (<I>) 
    {	
    	chomp;
	my ($w,@vector) = split(" ",$_);
	next if !defined($words{$w});
	print "$w @vector\n";
    }
    close(I);
}

sub loadV
{
    my ($file,$w2v) = @_;
    warn "\t\t-> loading word2vect <".basename($file)."> at --> ".localtime()."\n";
    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $!;
    my $index = 1;
    while (<I>) 
    {	
    	chomp;
	my ($w,$vector) = split(/\s+/,$_);
	$w2v->{$w} = $index;
	$index++;
    }
    close(I);
    foreach my $s (keys %prior)
    {
    	next if scalar keys %{$prior{$s}} <= 1;
    	my ($ssv,$m0th) = &mPaths();
    	open (STR, ">".$ssv) || die $!;
	binmode STR, ":utf8";
	my @P;
	foreach my $ent (keys %{$prior{$s}})
	{
	    $P[$kode{$s}{$ent}-1] = $prior{$s}{$ent};
	}
	print STR "@P\n";	    
    	close(STR);
    }
}

sub dodir
{
    my ($name) = @_;
    $name =~ s/\./#p#/g;
    $name =~ s/\$/#d#/g;
    $name =~ s/\(/#paz#/g;
    $name =~ s/\)/#pai#/g;
    $name =~ s/\&/#amp#/g;
    $name =~ s/\"/#kom0#/g;
    $name =~ s/\'/#kom1#/g;
    $name =~ s/\`/#kom2#/g;
    $name =~ s/\//#slh#/g;
    $name =~ s/\;/#pc#/g;
    $name =~ s/\:/#bp#/g;
    $name .= "###" if length($name) < 3;
    return $name;
}

sub ePaths
{
    my ($what) = @_;
    my $ent = &dodir($what);
    my $ssv; 
    my $f = substr($ent,0,1);
    my $s = substr($ent,1,1);
    my $t = substr($ent,2,1);
    $ssv = $path."ssv/train/entities/".$f."/".$s."/".$t."/".$ent.".w".$window;
    &createPaths($path."ssv/train/entities/",$f,$s,$t);
    return $ssv;
}

sub mPaths
{
    my $w = basename($DICT);
    $w =~ s/\.bz2//g;
    $w .= $tag;
    my $ssv; 
    my $m0th;
    if($TEST)
    {
	my $test = basename($TEST);
	$test =~ s/\.bz2//g;
	$test .= ".doc" if $DOC;
	$ssv = $path."ssv/test/".$w.".".$test.".".$nn;
	$ssv =~ s/\.py$//g;
	$m0th = $path."m\@ths/".$w.".".$nn;
	$m0th =~ s/\.py$//g;
    }
    if($CLUSTER || $TRAIN_M0THS)
    {
	$ssv = $path."ssv/train/".$w;
	$ssv =~ s/\.py$//g;
	$m0th = $path."m\@ths/".$w.".".$nn;
	$m0th =~ s/\.py$//g;
    }
    return ($ssv,$m0th);
}

sub createPaths
{
    my ($w,$f,$s,$t) = @_;
    mkdir $w."/".$f;
    mkdir $w."/".$f."/".$s;
    mkdir $w."/".$f."/".$s."/".$t;
}

sub loadS
{
    my ($file,$Prior,$entities,$Kode) = @_;
    warn "\t\t-> loading strings <".basename($file)."> at --> ".localtime()."\n" if (!$OUT);
    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $! ;
    while (<I>) 
    {	
	chomp;
	my ($w,@ents) = split(/\s/,$_) ;
	my $first = 0;
	my $T = 0;
	foreach my $en (@ents)
	{
	    my ($ent,$f)= &parse($en);
	    next if length($ent) >= 200; 
	    $first++;
	    $Prior->{$w}->{$ent} = $f;
	    $T += $Prior->{$w}->{$ent};
	    $entities->{$ent} = 1;
	    $Kode->{$w}->{$ent} = $first;
	}
	foreach my $en (@ents)
	{
	    my ($ent,$f)= &parse($en);
	    next if length($ent) >= 200; 
	    $Prior->{$w}->{$ent} /= $T;
	}
    }
    close(I);
}

sub loadSA
{
    my ($file,$Aux) = @_;
    warn "\t\t-> loading Aux strings <".basename($file)."> at --> ".localtime()."\n" if (!$OUT);
    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $! ;
    while (<I>) 
    {	
	chomp;
	my ($w,@ents) = split(/\s/,$_) ;
	my $first = 0;
	my $T = 0;
	foreach my $en (@ents)
	{
	    my ($ent,$f)= &parse($en);
	    next if length($ent) >= 200; 
	    $first++;
	    $Aux->{$w}->{$ent} = $f;
	    $T += $Aux->{$w}->{$ent};
	}
	foreach my $en (@ents)
	{
	    my ($ent,$f)= &parse($en);
	    next if length($ent) >= 200; 
	    $Aux->{$w}->{$ent} /= $T;
	}
    }
    close(I);
}

sub collect
{
    my ($file) = @_;
    foreach my $ent (keys %entities) #erase previous
    {
    	my $file = &ePaths($ent);
    	my $do = "rm ".$file;
    	my $out = system("$do") if (-e $file);
    	$file .= ".bz2";
    	$do = "rm ".$file;
    	$out = system("$do") if (-e $file);
    }
    warn "\t\t-> collecting entities <".basename($file)."> at --> ".localtime()."\n";
    my $d_kop = 0;
    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $! ;
    while (<I>) 
    {	
    	chomp;
    	my $doc = $_;
    	$d_kop++;
    	my @hitzak = split(" ",$doc);
    	for(my $x=0;$x < @hitzak; $x++)
    	{
    	    if($hitzak[$x] =~ /:::KEY/)
    	    {
    		my($anch,$ent,$key) = split(":::",$hitzak[$x]);
    		next unless defined($entities{$ent});
		my @teXt = ();	
		my $topea = 0;
		$topea = $x-$window if ($x-$window > 0);
		for(my $y=$x-1;$y >= $topea; $y--) #LEFT
		{
		    if ($hitzak[$y] =~ /:::KEY/) 
		    {
			my($a,$e,$k) = split(":::",$hitzak[$y]);
			my @words = split("_",$a);
			foreach my $tok (reverse(@words))
			{
			    $tok =~ s/\W$//g;
			    $tok =~ s/^\W//g;
			    unshift(@teXt,$tok);
			}
			$topea += @words-1 if @words >= 2;
		    }
		    else
		    {
			unshift(@teXt,$hitzak[$y]);
		    }
		}
		push(@teXt,"#m#".$anch."#m#");
		$topea = @hitzak;
		$topea = $x+1+$window if ($x+1+$window < @hitzak);
		for(my $y=$x+1;$y < $topea; $y++) #RIGHT
		{
		    if ($hitzak[$y] =~ /:::KEY/)
		    {
			my($a,$e,$k) = split(":::",$hitzak[$y]);
			my @words = split("_",$a);
			foreach my $tok (@words)
			{
			    $tok =~ s/\W$//g;
			    $tok =~ s/^\W//g;
			    push(@teXt,$tok);
			}
			$topea -= @words-1 if @words >= 2;
		    }
		    else
		    {
			push(@teXt,$hitzak[$y]);
		    }
		}
		&oWords(\@teXt,$anch);
		next unless @teXt;
		my $o = &printInstances($ent,\@teXt);
    	    }
    	}
    }
    close(I);
    warn "\ndone at -> ".localtime()."\n";
    warn "\n\t\t-> compressing entity files... \n";
    foreach my $ent (keys %entities)
    {
	my $ssv = &ePaths($ent);
	my $do = "bzip2 ".$ssv;
	my $out = system("$do") if (-e $ssv) && (-s $ssv);
    }
    warn "\ndone at -> ".localtime()."\n";
}

sub oWords
{
    my ($text,$anch) = @_;
    $anch =~ s/\_/ /g;
    $anch =~ s/\W$//g;
    $anch =~ s/^\W//g;
    my $quoted = quotemeta($anch);
    $anch =~ s/ /\_/g;
    my $raw = " @$text ";
    $raw =~ s/ $quoted / #o#$anch#o# /g;
    $raw =~ s/^ //g;
    $raw =~ s/ $//g;
    @$text = split(" ",$raw);
}

sub printInstances
{
    my ($e,$teXt) = @_;
    my ($ssv) = &ePaths($e);
    open (STR, ">>".$ssv) || die $!;
    binmode STR, ":utf8";
    print STR "@$teXt\n";
    close(STR);
    return 1;
}

sub cluster
{
    # foreach my $w (keys %prior) #erase previous
    # {
    # 	next if scalar keys %{$prior{$w}} <= 1;
    # 	my ($ssv,$m0th) = &mPaths();
    # 	$ssv .= ".bz2";
    # 	my $do = "rm ".$ssv;
    # 	my $out = system("$do") if (-e $ssv);
    # }
    warn "\t\t-> clustering entities at --> ".localtime()."\n";
    foreach my $w (keys %prior)
    {
	next if scalar keys %{$prior{$w}} <= 1;
	my ($ssv,$m0th) = &mPaths();
	open (STR, ">>".$ssv) || die $! ;
	binmode STR, ":utf8";
	foreach my $ent (keys %{$prior{$w}})
	{
	    my @smooth = (0) x ($window*2);
	    print STR $kode{$w}{$ent}." "."@smooth\n";	    
	    my $file = &ePaths($ent).".bz2";
	    print STR $kode{$w}{$ent}." "."@smooth\n" unless (-e $file);	    
	    next unless (-e $file);
	    my @same = ();
	    open (I, "-|:encoding(UTF-8)", "bzcat ".$file) || die $! ;
	    while (<I>) 
	    {	
		chomp;
		my $line = $_;
		push(@same,$line);
	    }
	    close(I);
	    my $top = 0;
	    foreach my $instance (shuffle(@same))
	    {	    
		my $context = &order($instance);
		next if @$context eq 0;
		$top++;
		last if $top eq $instance_size;
		print STR $kode{$w}{$ent}." "."@$context\n";
	    }
	    if($top < 1)
	    {
	    	my @smooth = (0) x ($window*2);
	    	print STR $kode{$w}{$ent}." "."@smooth\n";	    
	    }
	}    
	close(STR);
    }
    undef %w2v; #release memory
    warn "\ndone at -> ".localtime()."\n";
    warn "\n\t\t-> compressing mention files... \n";
    foreach my $w (keys %prior)
    {
	my ($ssv,$m0th) = &mPaths();
	my $do = "bzip2 ".$ssv;
	my $out = system("$do") if (-e $ssv) && (-s $ssv);
    }
    warn "\ndone at -> ".localtime()."\n";
}

sub order
{
    my ($text) = @_;
    $text = lc($text) unless $CASE;
    my ($left,$mention,$right) = split("#m#",$text);
    my @res;
    my @wordsL = split(" ",$left) if defined($left);
    my @wordsR = split(" ",$right) if defined($right);

    my $win = @wordsL;
    for (my $x=0; $x<=$win; $x++)
    {
	pop(@wordsL) if $x >= $window-1;
    }
    $win = @wordsR;
    for (my $x=0; $x<=$win; $x++)
    {
    	shift(@wordsR) if $x >= $window-1;
    }

    my @l;
    my @r;
    foreach my $w (@wordsL)
    {
	push(@l,$w2v{$w}) if defined($w2v{$w});
    }
    foreach my $w (@wordsR)
    {
	push(@r,$w2v{$w}) if defined($w2v{$w});
    }
    return \@res if (@l eq 0) and (@r eq 0);
    for (my $x=0; $x < $window-@l; $x++) #left padding
    {
	push (@res, 0);
    }
    push(@res,@l);
    push(@res,@r);
    for (my $x=0; $x < $window-@r; $x++) #right padding
    {
    	push (@res, 0);
    }
    return \@res;
}

sub trainM0ths
{
    warn "\t\t-> training m\@ths\n";

    my $plague = basename($DICT);
    $plague =~ s/\.bz2//g;
    $plague = $path."ssv/train/".$plague.$tag.".".$nn;
    $plague =~ s/\.py$//g;
    open (PLG, ">".$plague) || die $! ;
    binmode PLG, ":utf8";
    foreach my $w (keys %prior)
    {
	next if scalar keys %{$prior{$w}} <= 1;
	my ($ssv,$m0th) = &mPaths();
	print PLG $ssv.".bz2 ".$m0th."\n" if (-e $ssv.".bz2");
    }
    close(PLG);
    my $do = "python3 ".$nn." -t ".$plague." -v ".$W2V." > ".$plague.".deep";
    my $out = system("$do");
    warn "\ndone at -> ".localtime()."\n";
}

sub loadE
{
    my ($file) = @_;
    warn "\t\t-> testing m\@ths <".basename($file)."> at --> ".localtime()."\n";
    my @lines = ();
    my %erase = ();
    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $! ;
    while (<I>) 
    {	
	chomp;
	my($docid,$id,$o,$m,$mm,$hm,$type,$w_ctx,$m_ctx,$m_fd) = split (/\t/,$_); 
	my $w = &getW($m,$mm,$hm);                                                
	next if !defined($aux_dict{$w});
	next if scalar keys %{$aux_dict{$w}} <= 1;
	$erase{$w} = 1;
	my ($ssv,$m0th) = &mPaths();
	my $ctx;
	$ctx = &getAnchor($w_ctx,$w) unless $DOC; 
	$ctx = &getDoc($m_fd,$w) if $DOC; 
	open (STR, ">>".$ssv) || die $! ;
	binmode STR, ":utf8";
	foreach my $c (@$ctx)
	{
	    my $context;
	    $context = &order($c) unless $DOC;
	    $context = &shuffleDoc($c) if $DOC;
	    # @$context = (0) x ($window*2) if @$context eq 0;
	    next if @$context eq 0;
	    $erase{$w} = 0;
	    print STR $id.":::".@$ctx." @$context\n";
	}
	close(STR)
    }
    close(I);
    foreach my $w (keys %erase) 
    {
    	if ($erase{$w} eq 1)
    	{
    	    my ($ssv,$m0th) = &mPaths($w);
    	    my $do = "rm ".$ssv;
    	    my $out = system("$do") if (-e $ssv);		
    	}
    }
}

sub testM0ths
{
    my ($file) = @_;
    my %r = ();
    my %id2w = ();

    my $plague = basename($file);
    $plague =~ s/\.bz2//g;
    $plague = $path."ssv/test/".$plague.$tag.".".$nn;
    $plague =~ s/\.py$//g;
    open (PLG, ">".$plague) || die $! ;
    binmode PLG, ":utf8";
    foreach my $w (keys %prior)
    {
	my ($ssv,$m0th) = &mPaths();
	print PLG $ssv." ".$m0th."\n" if (-e $ssv) && (-e $m0th);
    }
    close(PLG);
    my $do = "python3 ".$nn." -r ".$plague." -v ".$W2V." > ".$plague.".deep";
    my $out = system("$do");
    undef %w2v; #release memory

    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $! ;
    while (<I>) 
    {	
	chomp;
	my $line = $_;
	my($docid,$id,$o,$m,$mm,$hm,$type,$w_ctx,$m_ctx,$m_fd) = split (/\t/,$line); 
	my $w = &getW($m,$mm,$hm);
	# my $w = "m\@thNster";
	foreach my $ent (keys %{$aux_dict{$w}}) #init all
	{
	    $r{$id}{$ent} = log($aux_dict{$w}{$ent});
	    $id2w{$id} = $w;
	}
	$r{$id}{"NIL"} = &nilProb();
    }	
    if(-e $plague.".deep")
    {
	open(PLG, $plague.".deep");
	binmode PLG, ":utf8";
	while(<PLG>)
	{
	    chomp;
	    my $line = $_;
	    my ($idun,@res) = split(" ",$line);
	    my ($id,$z) = split(":::",$idun);
	    my $w = $id2w{$id};
	    foreach my $ent (sort {$kode{"m\@thNster"}{$b} <=> $kode{"m\@thNster"}{$a}} keys %{$kode{"m\@thNster"}})
	    {
		my $k = $kode{"m\@thNster"}{$ent};
		my $prob = $res[$k-1];
		if (defined($aux_dict{$w}{$ent}))
		{
		    $r{$id}{$ent} += $prob/$z if defined($prob);
		    # $r{$id}{$ent} += log(1/$M)/$z unless defined($prob);
		    $r{$id}{$ent} -= log($aux_dict{$w}{$ent})/$z unless $PRIOR;
		}
	    }
	}
	close(PLG);
	open (PLG, ">".$plague.".deep") || die $! ;
    }

    open (I, "-|:encoding(UTF-8)", "bzcat $file") || die $! ;
    while (<I>) 
    {	
	chomp;
	my $line = $_;
	my($docid,$id,$o,$m,$mm,$hm,$type,$w_ctx,$m_ctx,$m_fd) = split (/\t/,$line); 
	my $w = &getW($m,$mm,$hm);
	print $docid."\t".$id."\t".$w."\t";
	foreach my $ent (sort {$r{$id}{$b} <=> $r{$id}{$a}} keys %{$r{$id}})
	{
	    next if $ent eq "NIL";
	    print $ent.":::".$r{$id}{$ent}." ";
	}
	print "NIL:::".$r{$id}{"NIL"}."\n";
    }
    close(I);
    warn "\ndone at -> ".localtime()."\n";
}

sub nilProb
{
    my $NIL = log(1/$M);
    $NIL += log(0.5) if $PRIOR;
    return $NIL;
}

sub parse 
{
    my ($str) = @_;
    my ($ent, $freq);
    $freq = 1;
    my @aux = split(/:/, $str);
    if (@aux > 1 && $aux[-1] =~ /\d+/) 
    {
	$freq = pop @aux;
    }
    $ent = join(":", @aux);
    $freq = 1 if $freq !~ /^[0-9,.Ee-]+$/; 
    return ($ent, $freq);
}

sub getDoc
{
    my ($text,$w) = @_;
    my @c;
    $text =~ s/#doc#//ig;
    for(my $i=0;$i<15;$i++)
    {
	push(@c,$text);
    }
    return \@c;
}

sub getAnchor
{
    my ($text,$w) = @_;
    my @c;
    $text =~ s/#m#/@@@@@/ig;
    $text =~ s/#o#/%%%%%/ig;
    while($text =~ /#ctx#[^#]+#\d+# ([^#]+) #ctx#[^#]+#\d+#/g)
    {
	my $anchor_text = $1;
	$anchor_text =~ s/@@@@@/#m#/g;
	$anchor_text =~ s/%%%%%/#o#/g;
	push(@c,$anchor_text);
    }
    push(@c,"") if @c eq 0;
    return \@c;  
}

sub getW
{
    my($m,$mm,$hm) = @_;
    my $n = $m; 
    $n =~ s/ /\_/g;
    $n = lc($n);                                  
    $n = lc($hm) if ($hm ne "NIL");     
    $n = lc($mm) if ($mm ne "NIL");     
    return $n;
}
