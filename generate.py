import sys
import copy
import itertools
from crossword import *


class CrosswordCreator:

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            length = len(word)
            for k in range(length):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or "_", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        # Determine grid size
        cell_size = 100
        img_width = self.crossword.width * cell_size
        img_height = self.crossword.height * cell_size

        # Create a blank image with white background
        image = Image.new("RGBA", (img_width, img_height), "white")
        draw = ImageDraw.Draw(image)

        # Load a font
        try:
            font = ImageFont.truetype("arial", 80)
        except IOError:
            font = ImageFont.load_default()

        # Draw grid
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [(j * cell_size, i * cell_size), ((j + 1) * cell_size, (i + 1) * cell_size)]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, outline="black", width=2)
                    if (i, j) in assignment.values():
                        text = assignment[i, j]
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        text_x = rect[0][0] + (cell_size - text_width) / 2
                        text_y = rect[0][1] + (cell_size - text_height) / 2
                        draw.text((text_x, text_y), text, fill="black", font=font)
                else:
                    draw.rectangle(rect, fill="black")

        image.save(filename)

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            length = len(word)
            for k in range(length):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack({})

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node consistent.
        (Remove any values that are inconsistent with a variable's unary
        constraints; in this case, the length of the word.)
        """
        for variable in self.domains:
            for word in set(self.domains[variable]):
                if len(word) != variable.length:
                    self.domains[variable].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value in `self.domains[y]`.
        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        if overlap:
            i, j = overlap
            for word_x in set(self.domains[x]):
                if not any(word_x[i] == word_y[j] for word_y in self.domains[y]):
                    self.domains[x].remove(word_x)
                    revised = True
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with all arcs in the problem.
        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = list(itertools.permutations(self.crossword.variables, 2))
        queue = arcs.copy()
        while queue:
            (x, y) = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((x, z))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        return set(assignment.keys()) == self.crossword.variables

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in the
        crossword puzzle without conflicting characters); return False
        otherwise.
        """
        for variable, word in assignment.items():

            # Check if all values are distinct
            if list(assignment.values()).count(word) > 1:
                return False

            # Check if the word matches the length of the variable
            if variable.length != len(word):
                return False

            # Check for conflicts with neighboring words
            for neighbor in self.crossword.neighbors(variable):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[variable, neighbor]
                    if word[i] != assignment[neighbor][j]:
                        return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list should be the one that rules out the
        fewest values among the neighbors of `var`.
        """
        def count_conflicts(value):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    for word in self.domains[neighbor]:
                        if word[j] != value[i]:
                            count += 1
            return count

        return sorted(self.domains[var], key=count_conflicts)

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values in
        its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned = [
            var for var in self.crossword.variables
            if var not in assignment
        ]

        def degree(var):
            return len(self.crossword.neighbors(var))

        return min(unassigned, key=lambda var: (len(self.domains[var]), -degree(var)))

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.
        `assignment` is a mapping from variables (keys) to words (values).
        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
