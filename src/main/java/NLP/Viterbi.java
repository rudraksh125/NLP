/**
 * @author kvivekanandan
 * Oct 25, 2015
 * Viterbi.java
 */

package NLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class Viterbi {

	public static List<List<Double>> viterbiMatrix = new ArrayList<List<Double>>();
	public static List<List<String>> backtrack = new ArrayList<List<String>>();

	public static List<String> maxLikelyStateSequence(List<String> input_observations, List<List<Double>> stateMatrix, List<List<Double>> emissionMatrix, Map<String, Integer> states, Map<String, Integer> observations, Map<String, Double> initStateProb) {
		viterbiMatrix = new ArrayList<List<Double>>();
		backtrack = new ArrayList<List<String>>();

		for (int i = 0; i < states.size() + 1; i++) {
			List<Double> list = new ArrayList<Double>(Collections.nCopies(input_observations.size() + 1, 1.0));
			viterbiMatrix.add(list);
		}
		for (int i = 0; i < states.size() + 1; i++) {
			List<String> list = new ArrayList<String>(Collections.nCopies(input_observations.size() + 1, ""));
			backtrack.add(list);
		}

		for (int s = 0; s < states.size(); s++) {
			double a = stateMatrix.get(0).get(s);
			String word = input_observations.get(0);
			if (observations.get(word) == null) {
				word = "UNK";
			}
			double b = emissionMatrix.get(s).get(observations.get(word));
			double log = Math.log10(a) + Math.log10(b);
			viterbiMatrix.get(s).set(0, log);
			backtrack.get(s).set(0, "");
		}

		for (int i = 1; i < input_observations.size(); i++) {
			String word = input_observations.get(i);
			for (int j = 0; j < states.size(); j++) {
				double max = Double.MAX_VALUE * -1;
				String backtrackS = "";
				for (int k = 0; k < states.size(); k++) {
					if (observations.get(word) == null) {
						word = "UNK";
					}
					double a = emissionMatrix.get(j).get(observations.get(word));
					double b = stateMatrix.get(k).get(j);
					double mu = viterbiMatrix.get(k).get(i - 1);
					double log = Math.log10(a) + Math.log10(b) + mu;
					if (log > max) {
						max = log;
						backtrackS = HMM.getKeyByValue(states, k);
					}
				}
				viterbiMatrix.get(j).set(i, max);
				backtrack.get(j).set(i, backtrackS);
			}
		}

		viterbiMatrix.get(states.size()).set(input_observations.size(), Double.MAX_VALUE);
		
		for (int s = 0; s < states.size(); s++) {
			String state = HMM.getKeyByValue(states, s);
			double a = stateMatrix.get(s).get(0);
			double likelihood = viterbiMatrix.get(s).get(input_observations.size() - 1) + Math.abs(Math.log10(a));
			if (likelihood < viterbiMatrix.get(states.size()).get(input_observations.size())) {
				viterbiMatrix.get(states.size()).set(input_observations.size(), likelihood);
				backtrack.get(states.size()).set(input_observations.size(), state);
			}
		}

		ArrayList<String> output = new ArrayList<String>();
		int hiddenStateIndex = states.size();
		int sIndex = input_observations.size();
		String tag = backtrack.get(hiddenStateIndex).get(sIndex);
		while (tag != null && tag.length() != 0) {
			String word = input_observations.get(sIndex - 1);
			output.add(word + "/" + tag);
			for (int i = 0; i < states.size(); i++) {
				if (HMM.getKeyByValue(states, i).equals(tag)) {
					hiddenStateIndex = i;
					break;
				}
			}
			tag = backtrack.get(hiddenStateIndex).get(--sIndex);
		}
		Collections.reverse(output);
		return output;

	}
}
