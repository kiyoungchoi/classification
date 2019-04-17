<?

use Phpml\Dataset\FilesDataset;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\Tokenization\WordTokenizer;
use Phpml\FeatureExtraction\StopWords\English;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Classification\SVC;
use Phpml\Metric\Accuracy;

require_once 'vendor/autoload.php';

ini_set('memory_limit','-1');

$dataset = new FilesDataset('bbc');

$split = new StratifiedRandomSplit($dataset, 0.3);
$samples = $split->getTrainSamples();

echo $samples[0];
exit;

$vectorizer = new TokenCountVectorizer(new WordTokenizer, new English());
$vectorizer->fit($samples);
$vectorizer->transform($samples);


$classifier = new SVC();
$classifier->train($samples, $split->getTrainLabels());


$testSamples = $split->getTestSamples();
$vectorizer->transform($testSamples);

$predicted = $classifier->predict($testSamples);



echo 'Accuracy: ' .Accuracy::score($split->getTestLabels(), $predicted);

echo '\n';

?>
