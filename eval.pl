use strict;
use warnings;
use Getopt::Long qw(:config auto_help); 
use File::Basename;

my ($TOP_C,$BOLD);
GetOptions("TOP_C=f" => \$TOP_C, "BOLD" => \$BOLD); 
die "Usage: $0 gs.key sys.key\n" unless @ARGV == 2;

my $gsFile = $ARGV[0];
my $sysFile = $ARGV[1];
my %gs;
my %sys;

my ($gsEznil) = 0;
my ($sysEznil) = 0;$sysEznil=0;
my ($okEznil) = 0;$okEznil = 0;
readGs();
readSys();

die if $okEznil eq 0;
$sysFile =~ s/%/%%/g;
# for my $id (keys %gs)
# {
#     print "$id not in sys\n" if !defined($sys{$id});
# }
if (!$BOLD)
{
    printf "| ".basename($sysFile)." \t| %2.2f |\n", 100*($okEznil)/($gsEznil) if $TOP_C;
    printf "| ".basename($sysFile)." -upperbound- \t| %2.2f |\n", 100*($okEznil)/($gsEznil) if !$TOP_C;
}
else
{
    printf "| *".basename($sysFile)."* \t| *%2.2f* |\n", 100*($okEznil)/($gsEznil) if $TOP_C;
    printf "| *".basename($sysFile)."* -upperbound- \t| *%2.2f* |\n", 100*($okEznil)/($gsEznil) if !$TOP_C;
}

sub readGs
{
    open(Fitx,"$gsFile") || die;
    binmode Fitx, ":utf8";
    while (<Fitx>)
    {
	chomp;
	my($doc,$id,$offset,$mention,$ent,@rest) = split("\t",$_);
	$gs{$id}=$ent if ($ent !~ /NIL/);
	$gsEznil++  if ($ent !~ /NIL/); 
    }
    close(Fitx);
}

sub readSys
{
    open(Fitx,"$sysFile") || die;
    binmode Fitx, ":utf8";
    while (<Fitx>)
    {
	chomp;
	my($doc,$id,$name,$ents) = split("\t",$_);
	# $sys{$id} = 1;
	my @e = split(" ",$ents);
	my $top = 0;
	my $not_found = 1;
	foreach my $en (@e)
	{
	    last if defined($TOP_C) && $top eq $TOP_C;
	    next if !defined($gs{$id});
	    my ($ent,$p,@rest) = split(":::",$en);
	    $okEznil++ if $ent eq $gs{$id};
	    $not_found = 0 if $ent eq $gs{$id};
	    $top++;
	}
	#warn "noCG $id --> name:$name entity:".$gs{$id}."\n" if $not_found eq 1 && defined($gs{$id});
    }
    close(Fitx);
}
