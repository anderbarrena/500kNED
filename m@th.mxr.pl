use strict;
use warnings;
use Getopt::Long qw(:config auto_help); 

my ($COEF,$DICT,$ALL_RANKS,$INV,$KB,$NIL);
GetOptions("COEF=s" => \$COEF, "DICT=s" => \$DICT, "ALL_RANKS" => \$ALL_RANKS, "INV" => \$INV, "KB=s" => \$KB, "NIL" => \$NIL);
my @coefs;

binmode STDOUT, ":utf8";
binmode STDIN, ":utf8";

if(defined($COEF))
{
    @coefs = split("-",$COEF);
    die "Coef and Args do not match.\n" if (@coefs ne @ARGV);
    my $tot = 0;
    foreach my $cof (@coefs)
    {
	$tot += $cof;
    }
    foreach my $cof (@coefs)
    {
	$cof/= $tot;
    }
}

my %kb = ();
my %d = ();
my %cmb = ();
my %e_to_rest = ();
my %subset = ();

&read(\@ARGV);
&read_dict($DICT) if $DICT;
&read_kb($KB) if $KB;
&mix_log();

sub read
{
    my $arg = $_[0];
    my $cf = 0;
    foreach my $file (@$arg)
    {
	open(Fitx,"$file") || die;
	binmode Fitx, ":utf8";
	while (<Fitx>)
	{
	    chomp;
	    my($doc,$id,$name,$ents,$flag) = split("\t",$_);
	    foreach my $e_c (split(" ",$ents))
	    {
		$e_c =~ s/::::/:::/g;
		my($e,$c,$rest) = split(":::",$e_c);
		$c *= $coefs[$cf] if defined($COEF);
		$cmb{$doc}{$id}{$name}{$e} -= $c if $INV;
		$cmb{$doc}{$id}{$name}{$e} += $c if !$INV;
		$e_to_rest{$doc}{$id}{$name}{$e} = $rest if defined($rest);
		$subset{$name}++;
	    }
	}
	$cf++;
	close(Fitx);
    }
}

sub mix_log
{
    foreach my $doc (keys %cmb)
    {
	foreach my $id (keys %{$cmb{$doc}})
	{
	    foreach my $name (keys %{$cmb{$doc}{$id}})
	    {
		print $doc."\t".$id."\t".$name."\t";
		foreach my $e (sort {$cmb{$doc}{$id}{$name}{$b} <=> $cmb{$doc}{$id}{$name}{$a}} keys %{$cmb{$doc}{$id}{$name}})
		{
		    next if $e eq "NIL";
		    next if &dict($id,$name,$e) && $DICT;
		    next if &heuristics($e);
		    next if !defined($kb{$e}) && ($KB && !$NIL);
		    my $prob = $cmb{$doc}{$id}{$name}{$e};
		    $e = "NIL_".$e if !defined($kb{$e}) && $KB && $NIL;
		    print $e.":::".$prob." " if !defined($e_to_rest{$doc}{$id}{$name}{$e});
		    print $e.":::".$prob.":::".$e_to_rest{$doc}{$id}{$name}{$e}." " if defined($e_to_rest{$doc}{$id}{$name}{$e});
		    last if !$ALL_RANKS;
		}
		my $prob = $cmb{$doc}{$id}{$name}{"NIL"};
		print "NIL".":::".$prob."\n" if !defined($e_to_rest{$doc}{$id}{$name}{"NIL"});
		print "NIL".":::".$prob.":::".$e_to_rest{$doc}{$id}{$name}{"NIL"}."\n" if defined($e_to_rest{$doc}{$id}{$name}{"NIL"});
	    }
	}
    }
}

sub dict
{
    my $id = shift;
    my $s = shift;
    my $e = shift;
    my $jump = 1;
    if(!defined($d{$s}))
    {
	$jump = 0;
    }
    elsif(defined($d{$s}{$e}))
    {
	$jump = 0;
    }
    return $jump;
}

sub heuristics
{
    my $e = shift;
    my $jump = 0;
    $jump = 1 if $e =~ /disambiguation/;
    $jump = 1 if $e =~ /list\_of/i;    
    $jump = 1 if $e eq "Newline";
    return $jump;
}

sub read_dict 
{
    open (I, "-|:encoding(UTF-8)", "bzcat $_[0]") || die $! ;
    while (<I>) 
    {
	chomp;
	my ($w,@ents) = split(/\s/,$_) ;
	foreach my $ent (@ents)
	{
	    my($ent,$f)= &parse_ent_prob($ent);
	    $d{$w}{$ent}++ if defined($subset{$w});
	}
    }
    close(I);
}

sub read_kb 
{
    open (I, "-|:encoding(UTF-8)", "bzcat $_[0]") || die $! ;
    while (<I>) 
    {
	chomp;
	$kb{$_}=1;
    }
    close(I);
}

sub parse_ent_prob 
{
    my $str = shift;
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
