import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
import pandas as pd
from ingestion.entity_resolver import EntityResolver
from shared.data_classes import EntityLabels


resolver = EntityResolver(
    kg_entities=None,
    merge_threshold=0.88,
    cross_doc_threshold=0.82,
    use_ner_with_confidence=True,
    use_cot=True)
text = "Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator .He was born in the city of Loja .He is best known today as the author of Ecuador 's first novel La Emancipada ( 1863 ) .Owing to the book 's length , usually less than 100 pages long , many experts have argued that it is really a novella rather than a full novel , and that Ecuador 's first novel is Juan León Mera 's Cumanda ( 1879 ) .Nevertheless , thanks to the arguments of the well - known and respected Ecuadorian writer Alejandro Carrión ( 1915 – 1992 ) , Miguel Riofrío 's La Emancipada has been accepted as Ecuador 's first novel .Riofrio died in exile in Peru ."
# chunk_info, clean_df, review_df = resolver.process_document(text)

# resolver.save_full_state(settings.ENTITY_RESOLUTION_STATE_PATH, clean_df=clean_df, review_df=review_df)

resolver.add_user_resolution(
        canonical_name="Miguel Riofrio Sánchez",
        entity_label=EntityLabels.PER,
        summary="Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran poet , novelist , journalist , orator , and educator",
        aliases=["Miguel Riofrio", "Miguel Riofrío"],
        context_indicators=["journalist", "novelist", "orator", "educator", "Ecuadoran poet"],
    )

# row 1: Ecuador (was wrongly -> Daniel Noboa)
resolver.add_user_resolution(
    canonical_name="Ecuador",
    entity_label=EntityLabels.LOC,
    summary="Ecuador,[a] officially the Republic of Ecuador,[b] is a country in northwestern South America, bordered by Colombia on the north, Peru on the east and south, and the Pacific Ocean on the west. It also includes the Galápagos Province which contains the Galápagos Islands in the Pacific, about 1,000 kilometers (540 nmi; 620 mi) west of the mainland. The country's capital is Quito, and the largest city is Guayaquil.",
    context_indicators=["country"],
)

# row 2: Loja (was wrongly -> Libertad F.C.)
resolver.add_user_resolution(
    canonical_name="Loja Ecuador",
    entity_label=EntityLabels.LOC,
    summary="Loja is the capital of Ecuador's Loja Province. It is located in the Cuxibamba valley in the south of the country, sharing borders with the provinces of Zamora-Chinchipe and other cantons of the province of Loja. Loja holds a rich tradition in the arts, and for this reason is known as the Music and Cultural Capital of Ecuador. The city is home to two major universities.",
    aliases=["Loja"],
    context_indicators=["capital of Ecuador", "located in south of the country", "located in the Cuxibamba valley"],
)

resolver.add_user_resolution(
    canonical_name="peru",
    entity_label=EntityLabels.LOC,
    summary="Peru is a country in South America that's home to a section of Amazon rainforest and Machu Picchu, an ancient Incan city high in the Andes mountains. The region around Machu Picchu, including the Sacred Valley, Inca Trail and colonial city of Cusco, is rich in archaeological sites. On Peru’s arid Pacific coast is Lima, the capital, with a preserved colonial center and important collections of pre-Columbian art.")

resolver.add_user_resolution(
    canonical_name="La Emancipada",
    entity_label=EntityLabels.MISC,
    summary="La Emancipada is a novel by the Ecuadorian writer Miguel Riofrío, published in 1863. It is considered the first novel published in Ecuador. The story critiques machismo, the power of the Church, and authoritarianism within the family through the tragic fate of a young woman.")


resolver.add_user_resolution(
    canonical_name="Juan León Mera",
    entity_label=EntityLabels.PER,
    summary="Juan León Mera Martínez was an Ecuadorian essayist, novelist, politician and painter. His best-known works are the Ecuadorian National Hymn and the novel Cumandá. Additionally, in his political career, he was a functionary of president Gabriel García Moreno")

resolver.add_user_resolution(
    canonical_name="Alejandro Carrión",
    entity_label=EntityLabels.PER,
    aliases=["Carrión"],
    summary="Alejandro Carrión Aguirre was an Ecuadorian poet, novelist and journalist. He wrote the novel La espina, the short story book La manzana dañada, and numerous poetry books. As a journalist he published many of his articles under the pseudonym 'Juan Sin Cielo.'")


resolver.add_user_resolution(
    canonical_name="Cumandá",
    entity_label=EntityLabels.PER,
    aliases=["Cumanda"],
    summary="The term Cumandá most notably refers to the classic 1879 foundational Ecuadorian novel by Juan León Mera, a geographic canton in Ecuador, and an herbal health tincture",
    context_indicators=["romantic novel", "novel"],
    related_to=["Juan León"]
    )



chunk_info, clean_df, review_df = resolver.process_document(text)

resolver.save_full_state(settings.ENTITY_RESOLUTION_STATE_PATH, clean_df=clean_df, review_df=review_df)
resolver.load_full_state(settings.ENTITY_RESOLUTION_STATE_PATH)