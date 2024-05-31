import csv
import itertools
import sys

PROBS = {
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    "trait": {
        2: {
            True: 0.65,
            False: 0.35
        },
        1: {
            True: 0.56,
            False: 0.44
        },
        0: {
            True: 0.01,
            False: 0.99
        }
    },
    "mutation": 0.01
}

def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if known, blank otherwise.
    """
    data = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data

def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.
    """
    probability = 1

    for person in people:
        genes = (
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )

        mother = people[person]["mother"]
        father = people[person]["father"]

        # Probability of gene
        if mother is None and father is None:
            probability *= PROBS["gene"][genes]
        else:
            # Probability from parents
            mother_genes = (
                2 if mother in two_genes else
                1 if mother in one_gene else
                0
            )
            father_genes = (
                2 if father in two_genes else
                1 if father in one_gene else
                0
            )

            # Probabilities of passing the gene
            pass_from_mother = (
                1 - PROBS["mutation"] if mother_genes == 2 else
                0.5 if mother_genes == 1 else
                PROBS["mutation"]
            )
            pass_from_father = (
                1 - PROBS["mutation"] if father_genes == 2 else
                0.5 if father_genes == 1 else
                PROBS["mutation"]
            )

            if genes == 2:
                probability *= pass_from_mother * pass_from_father
            elif genes == 1:
                probability *= (pass_from_mother * (1 - pass_from_father) +
                                (1 - pass_from_mother) * pass_from_father)
            else:
                probability *= (1 - pass_from_mother) * (1 - pass_from_father)

        # Probability of trait
        probability *= PROBS["trait"][genes][person in have_trait]

    return probability

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value to update depends on whether they are in `one_gene`, `two_genes`, and `have_trait`.
    """
    for person in probabilities:
        genes = (
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )
        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][person in have_trait] += p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution is normalized
    (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        for field in ["gene", "trait"]:
            total = sum(probabilities[person][field].values())
            for value in probabilities[person][field]:
                probabilities[person][field][value] /= total

def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have one copy of the gene
        for one_gene in powerset(names - have_trait):

            # Loop over all sets of people who might have two copies of the gene
            for two_genes in powerset(names - have_trait - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in ["gene", "trait"]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")

if __name__ == "__main__":
    main()

