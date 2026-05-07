"""Generate example PDF from Federalist Papers excerpts."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY

# Federalist Papers excerpts (public domain)
CONTENT = [
    ("The Federalist Papers: An Introduction", """
The Federalist Papers are a collection of 85 essays written to promote the ratification of the
United States Constitution. Published between October 1787 and May 1788, these essays examine
the benefits of the new federal system and address concerns raised by opponents of the Constitution.
The authors, Alexander Hamilton, James Madison, and John Jay, wrote under the pseudonym Publius
to encourage thoughtful debate about this important new framework for American government.
"""),
    ("On the Union", """
It has been frequently remarked that it seems to have been reserved to the people of this country,
by their conduct and example, to decide the important question, whether societies of men are really
capable or not of establishing good government from reflection and choice, or whether they are forever
destined to depend for their political constitutions on accident and force. The Federalist advocates
for a strong, well-ordered union under the proposed Constitution.
"""),
    ("Separation of Powers", """
The accumulation of all powers, legislative, executive, and judiciary, in the same hands, whether
of one, a few, or many, and whether hereditary, self-appointed, or elective, may justly be pronounced
the very definition of tyranny. Therefore, the Constitution carefully divides governmental power among
three distinct branches. This separation ensures that no single branch becomes too powerful and protects
the liberty of the people.
"""),
    ("Checks and Balances", """
Ambition must be made to counteract ambition. The interest of the man must be connected with the
constitutional rights of the place. It may be a reflection on human nature, that such devices should
be necessary to control the abuses of government; but what is government itself, but the greatest of
all reflections on human nature? If men were angels, no government would be necessary.
"""),
    ("The Federalist System", """
The proposed Constitution is, in strictness, neither a national nor a federal Constitution, but a
composition of both. In its foundation it is federal, not national; in the sources from which the
ordinary powers of the government are drawn, it is partly federal and partly national; in the operation
of these powers, it is national, not federal; in the extent of them, again, it is federal, not national.
"""),
    ("The Legislative Branch", """
The legislative authority necessarily predominates. The remedy for this inconvenience is to divide the
legislature into different branches; and to render them, by different modes of election and different
principles of action, as little connected with each other as the nature of their common functions and
their common dependence on the society will admit. The House and Senate provide essential checks on
legislative power.
"""),
    ("The Executive Power", """
The executive power is generally stronger in proportion as its unity is greater. There is an obvious
necessity for that energy and despatch which a single executive can enjoy and which is essential to
the protection of the community against foreign attacks. Yet the executive is carefully constrained by
the legislative and judicial branches through the system of checks and balances.
"""),
    ("Judicial Review", """
Whoever attentively considers the different departments of power must perceive, that, in a government
in which they are separated from each other, the judiciary, from the nature of its functions, will always
be the least dangerous branch to the political rights of the Constitution; because it will be least in
a capacity to annoy or injure them. The courts interpret the law and determine whether government
actions comply with the Constitution.
"""),
    ("The Commerce Clause", """
The regulation of commerce is a matter of national concern. The power given to Congress in the Clause
to regulate interstate and foreign commerce grants the federal government authority over trade between
states and with foreign nations. This power was essential to create a unified national economy and prevent
destructive trade wars between the states.
"""),
    ("The Bill of Rights", """
The conventions of a number of the States, having at the time of their adopting the Constitution,
expressed a desire, in order to prevent misconstruction or abuse of its powers, that further declaratory
and restrictive clauses should be added. In response to these concerns, the Bill of Rights was added
to the Constitution to protect fundamental individual liberties including freedom of speech, religion,
press, and assembly.
"""),
    ("Federalism and State Sovereignty", """
The powers not delegated to the United States by the Constitution, nor prohibited by it to the States,
are reserved to the States respectively, or to the people. This principle of federalism allows states
to retain significant powers while granting the federal government authority over matters of national
concern. This division of power protects local liberty and allows for diversity in governance.
"""),
    ("Amendment and Change", """
The mode in which the Constitution came into being established this principle. The Constitution was
not established by an act of one state, but by the people of the different states uniting for a common
purpose. Provision for amendment ensures that the Constitution can evolve to meet changing needs while
maintaining its fundamental structure and protections for liberty.
"""),
]

def make_example_pdf(output_path: str = "examples/knowledge_base/example.pdf") -> None:
    """Generate example PDF.

    Args:
        output_path: Output path for PDF.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Create custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor="#000000",
        spaceAfter=12,
        alignment=TA_JUSTIFY,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor="#1a1a1a",
        spaceAfter=6,
        spaceBefore=12,
        alignment=TA_JUSTIFY,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
    )

    # Add title
    story.append(Paragraph("The Federalist Papers", title_style))
    story.append(Paragraph("A Comprehensive Guide to the American Constitution", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Add content sections
    for title, content in CONTENT:
        story.append(Paragraph(title, heading_style))
        story.append(Paragraph(content.strip(), body_style))

        # Add page breaks every 3 sections
        if CONTENT.index((title, content)) % 3 == 2:
            story.append(PageBreak())

    # Build PDF
    doc.build(story)
    print(f"Generated example PDF: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    output = sys.argv[1] if len(sys.argv) > 1 else "examples/knowledge_base/example.pdf"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    make_example_pdf(output)
