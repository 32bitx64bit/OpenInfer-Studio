package quantize

import (
	"fmt"
	"strings"
)

func pick(list []string, i, stride int) string {
	if len(list) == 0 {
		return ""
	}
	if stride < 1 {
		stride = 1
	}
	return list[(i/stride)%len(list)]
}

func buildGeneratedCalibrationCorpus() calibrationCorpus {
	places := synthPlaces()
	objects := synthObjects()
	actions := synthActions()
	qualities := synthQualities()
	units := synthUnits()
	weathers := synthWeathers()
	materials := synthMaterials()
	crafts := synthCrafts()
	faults := synthFaults()

	var corpus calibrationCorpus
	for i := 0; i < syntheticCalibrationRecords; i++ {
		id := fmt.Sprintf("openinfer-synthetic-v2-%04x", i)
		scene := synthScene{
			place:    pick(places, i, 1),
			object:   pick(objects, i, len(places)),
			action:   pick(actions, i, 7),
			quality:  pick(qualities, i, 13),
			unit:     pick(units, i, 17),
			marker:   alphaMarker(i),
			weather:  pick(weathers, i, 19),
			material: pick(materials, i, 23),
			craft:    pick(crafts, i, 29),
			fault:    pick(faults, i, 31),
			index:    i,
		}
		domain := syntheticCalibrationDomains[i%len(syntheticCalibrationDomains)]
		text := buildSyntheticCalibrationText(domain, scene, places, objects, qualities, units)
		partition := calibrationRecordPartition(id)
		appendCalibrationRecord(&corpus, calibrationRecord{
			ID:        id,
			Domain:    domain,
			Source:    "OpenInfer deterministic synthetic corpus v2 (project-authored)",
			Partition: partition,
			Text:      text,
		})
	}
	topUpDomainHoldouts(&corpus)
	interleaveCalibrationCorpus(&corpus)
	return corpus
}

func buildSyntheticCalibrationText(domain string, s synthScene, places, objects, qualities, units []string) string {
	var text string
	switch domain {
	case "facts":
		text = factsCalibrationRecord(s)
	case "code":
		text = codeCalibrationRecord(s)
	case "multilingual":
		text = multilingualCalibrationRecord(s)
	case "structured-tool":
		text = structuredToolCalibrationRecord(s)
	case "long-context":
		text = longContextCalibrationRecord(s, places, objects, qualities, units)
	case "refusal-adjacent":
		text = refusalCalibrationRecord(s)
	case "chat":
		text = chatCalibrationRecord(s)
	default:
		text = proseCalibrationRecord(s)
	}
	if domain != "long-context" {
		text = extendRecordIfShort(text, s)
	}
	text += colorCalibrationNote(s)
	return text
}

func synthPlaces() []string {
	return []string{
		"alder quay", "basalt ridge", "cedar workshop", "delta observatory", "elm archive", "frost meadow",
		"granite harbor", "hazel orchard", "iron footbridge", "juniper station", "karst valley", "larch library",
		"maple lock", "north jetty", "oak mill", "pine trestle", "quartz adit", "reed marsh",
		"slate kiln", "tide gauge", "umber quarry", "violet boathouse", "willow weir", "xenolith cut",
		"yellow beacon", "zinc depot", "amber causeway", "brine cistern", "copper loft", "drift fence",
		"eddy pool", "flint bench", "gypsum shelf", "heather saddle", "inlet stair", "jasper spur",
		"kelp yard", "limestone nick", "mist landing", "needle pass", "oxbow camp", "pebble levee",
		"quince terrace", "rime gallery", "silt basin", "thatch loft", "undercliff path", "vein house",
	}
}

func synthObjects() []string {
	return []string{
		"barometer", "compass", "drain gauge", "field notebook", "gear train", "humidity sensor",
		"index card", "junction box", "kiln timer", "level staff", "map case", "network probe",
		"anemometer", "burette", "clinometer", "dipstick", "eddy meter", "flowmeter",
		"goniometer", "hydrometer", "inclinometer", "joulemeter", "kymograph", "lux meter",
		"micrometer", "needle valve", "oscilloscope", "pyrometer", "quadrant", "rheostat",
		"sextant", "tachometer", "u-tube", "voltmeter", "wattmeter", "xy recorder",
		"altimeter", "bolometer", "calorimeter", "densitometer", "ellipsometer", "fluorometer",
		"galvanometer", "hygrometer", "interferometer", "manometer", "nephometer", "potentiometer",
	}
}

func synthActions() []string {
	return []string{
		"audited", "balanced", "cataloged", "decoded", "estimated", "filtered",
		"grouped", "measured", "normalized", "reconciled", "sampled", "verified",
		"aligned", "bracketed", "collated", "dated", "enumerated", "folded",
		"hashed", "indexed", "logged", "mirrored", "notarized", "offset",
		"parsed", "quoted", "ranked", "sealed", "tagged", "unspooled",
		"weighed", "zeroed", "annotated", "batched", "copied", "drafted",
	}
}

func synthQualities() []string {
	return []string{
		"amber", "brisk", "copper", "dry", "even", "faint",
		"granular", "hollow", "irregular", "level", "muted", "steady",
		"ashen", "bleak", "crisp", "damp", "eerie", "foggy",
		"gilt", "harsh", "icy", "jaunty", "keen", "limp",
		"mild", "narrow", "overcast", "pale", "quiet", "raw",
		"sleet", "thin", "uneven", "violet", "warm", "zinc-grey",
	}
}

func synthUnits() []string {
	return []string{
		"bytes", "centimetres", "degrees", "grams", "hertz", "joules",
		"kilometres", "litres", "minutes", "newtons", "pascals", "seconds",
		"amperes", "candelas", "kelvin", "moles", "ohms", "teslas",
		"watts", "coulombs", "farads", "henries", "lux", "siemens",
		"becquerels", "grays", "sieverts", "webers", "katals", "radians",
	}
}

func synthWeathers() []string {
	return []string{
		"sleet", "drizzle", "hail", "fog", "glare", "gusts",
		"hoarfrost", "haze", "katabatic wind", "lee eddy", "mackerel sky", "northeaster",
		"open sun", "pack-ice smell", "quiet thaw", "rime", "sea smoke", "thunderhead",
		"undercloud", "virga", "whiteout edge", "yellow dusk", "zephyr", "black ice",
		"graupel", "sun dog", "freezing drizzle", "dry lightning", "ground blizzard", "chinook",
	}
}

func synthMaterials() []string {
	return []string{
		"oak", "brass", "slate", "linen", "ceramic", "graphite",
		"hemp", "ivory paper", "jute", "krypton glass", "lead", "mahogany",
		"nickel", "obsidian", "pewter", "quartz", "resin", "steel",
		"teak", "umber glaze", "varnish", "waxed cloth", "yarn", "zinc",
		"alder", "bronze", "cork", "felt", "horn", "ivory-free bone",
	}
}

func synthCrafts() []string {
	return []string{
		"surveying", "bookbinding", "horology", "optics", "cartography", "metallurgy",
		"navigation", "pottery", "printmaking", "rigging", "stonecutting", "taxonomy",
		"weaving", "woodturning", "assay", "cooperage", "dyeing", "engraving",
		"fletching", "gilding", "hydrology", "illumination", "joinery", "knapping",
		"lapidary", "millwrighting", "cooperage-repair", "ropework", "saddlery", "tinsmithing",
	}
}

func synthFaults() []string {
	return []string{
		"a skipped checksum", "a reversed sign", "a dropped unit", "a truncated timestamp",
		"a swapped pair", "a silent overflow", "a sticky relay", "a fogged lens",
		"a warped staff", "a frayed cable", "a missed carry", "a stale cache",
		"an off-by-one index", "a fused contact", "a smeared glyph", "a late echo",
		"a hung mutex", "a bent needle", "a wet page", "a crossed polarity",
		"a missing modulus", "a clipped peak", "a ghost duplicate", "an unsigned wrap",
		"a double-counted row", "a timezone-naive stamp", "a swapped endian word", "a stale zero",
		"a torn checksum trailer", "a misaligned decimal",
	}
}

func synthDetails() []string {
	return []string{
		"lichen", "kestrel", "rowan", "clinker", "thimble", "marl",
		"wainscot", "gillnet", "spindle", "cowrie", "tinder", "quoin",
		"hawthorn", "capstan", "punt", "scree", "whinstone", "eelgrass",
		"linseed", "gimlet", "creel", "peat", "adze", "wattle",
		"bramble", "cleat", "dory", "flintknap", "gorse", "hame",
		"inkstone", "jetty-ring", "kelson", "lanyard", "millstone", "nettle",
		"oarlock", "pintle", "quern", "rowlock", "sailcloth", "tiller",
		"umbel", "verdigris", "whetstone", "yardarm", "zostera", "anvil-block",
		"barleycorn", "caulk", "drawknife", "eddy-line", "fid", "grommet",
		"handspike", "ibis", "jib-sheet", "kedge", "leister", "mast-hoop",
		"nock", "oakum", "parrel", "quarry-sap", "ratline", "swingle",
		"treenail", "upright-loom", "vang", "woolding", "xylograph", "yarrow",
		"aspen", "bilge", "cringle", "deadeye", "ell", "felloe",
		"gudgeon", "hackle", "inlet-pool", "joggle", "kevel", "limpet",
		"mizzen", "nocking-point", "osier", "pauldron", "quillon", "rowen",
		"sennit", "thole", "uvula-of-bell", "vetch", "windlass", "yawl",
		"alehouse-bench", "binnacle", "cobble", "dulse", "ember-pan", "firkin",
		"gasket-cord", "heather-bell", "ice-rime", "juniper-berry", "kilderkin", "luff",
		"marline", "needfire", "otter-spraint", "plimsoll-mark", "quoin-key", "reedmace",
		"starling-of-pier", "tumblehome", "understory", "vane-card", "washstrake", "yellowhammer",
		"alderkin", "brackish-pool", "clough", "drift-pin", "esker", "ford-stone",
		"glyptic-chip", "holt", "isthmus-path", "jet-bead", "knoll", "loess",
		"moraine", "ness", "oxbow-mud", "polder-gate", "quintain", "riffle",
		"shingle-bank", "tarn", "upland-kine", "vale-mist", "weir-pool", "zetetic-note",
	}
}

func colorCalibrationNote(s synthScene) string {
	details := synthDetails()
	a := pick(details, s.index, 1)
	b := pick(details, s.index, 7)
	c := pick(details, s.index, 13)
	switch s.index % 5 {
	case 1:
		return fmt.Sprintf("\nUser: Margin doodle for %s?\nAssistant: A %s beside a %s, sketched in %s weather. It is not a reading in %s and does not repair %s.\n", s.marker, a, b, s.weather, s.unit, s.fault)
	case 2:
		return fmt.Sprintf("\nUser: What object in the margin of %s is ignorable?\nAssistant: The %s. It sits near %s furniture and a %s. Ignore it when copying unit %s at %s.\n", s.marker, c, s.material, a, s.unit, s.place)
	case 3:
		return fmt.Sprintf("\nUser: Inventory the non-instrument clutter for %s.\nAssistant: %s, %s, and a leftover %s from %s work. None of them convert %s.\n", s.marker, a, b, c, s.craft, s.unit)
	case 4:
		return fmt.Sprintf("\nUser: One sensory aside for log %s, not a measurement.\nAssistant: Smell of %s, glint of %s, pocket %s. Quality %s stays an adjective. Marker %s.\n", s.marker, a, b, c, s.quality, s.marker)
	default:
		return fmt.Sprintf("\nUser: Why is there a %s in the photo of %s?\nAssistant: It is clutter. The %s and the %s are equally irrelevant to the %s in %s. Do not copy them into the ledger.\n", a, s.marker, b, c, s.object, s.unit)
	}
}

func proseCalibrationRecord(s synthScene) string {
	switch (s.index / len(syntheticCalibrationDomains)) % 8 {
	case 1:
		return proseIncidentRecord(s)
	case 2:
		return proseDiaryRecord(s)
	case 3:
		return proseLetterRecord(s)
	case 4:
		return proseSafetyRecord(s)
	case 5:
		return proseInventoryRecord(s)
	case 6:
		return proseOralHistoryRecord(s)
	case 7:
		return proseWorkshopRecord(s)
	default:
		return proseFieldNoteRecord(s)
	}
}

func proseFieldNoteRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Summarize field note %s without inventing a result. Include how the observer handled a disagreement between the %s and the ledger, and what was left unresolved. Weather was %s; the housing is %s.\n", s.marker, s.object, s.weather, s.material)
	fmt.Fprintf(&b, "Assistant: At the %s, the observer %s the %s under %s conditions. The %s reading was copied into a separate ledger, checked against the instrument label, and left unresolved when the two sources disagreed. Note %s does not guess a missing value; it only records that the label and the ledger were not the same page. The weather note stays qualitative (%s) so it cannot be mistaken for a converted quantity. The %s craft of the bench is mentioned only as furniture, not as a measurement.\n", s.place, s.action, s.object, s.quality, s.unit, s.marker, s.weather, s.craft)
	fmt.Fprintf(&b, "User: If a later editor wants a single headline for note %s, what is allowed and what is not?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Allowed: \"%s: %s at %s, unit %s, disagreement recorded.\" Not allowed: filling the gap with an average, changing %s into another unit, or dropping marker %s from the filename. The headline is a pointer, not a new measurement. Do not blame %s in the headline; that is a separate defect log.\n", s.marker, s.object, s.place, s.unit, s.unit, s.marker, s.fault)
	fmt.Fprintf(&b, "User: Write two sentences a night-shift reader can use if they only have the headline for %s.\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Open the ledger row whose header is %s and compare it to the instrument tag at %s. If they disagree, copy both numbers with the unit %s and stop; do not %s a substitute from memory. The %s stays the subject of the row.\n", s.marker, s.place, s.unit, s.action, s.object)
	return b.String()
}

func proseIncidentRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: File an incident report for ticket %s. The %s at %s disagreed with the night ledger. Suspected defect: %s. Do not invent a corrected reading.\n", s.marker, s.object, s.place, s.fault)
	fmt.Fprintf(&b, "Assistant: Incident %s. Location %s. Instrument: %s in a %s housing. Observer %s the device under %s conditions during %s. The ledger and the label disagreed; both values remain on the page with unit %s. Defect note: %s. No substitute number was written. Weather %s is environment, not a reading.\n", s.marker, s.place, s.object, s.material, s.action, s.quality, s.craft, s.unit, s.fault, s.weather)
	fmt.Fprintf(&b, "User: Who is allowed to close ticket %s?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Only a second observer who sees both original numbers. Closing by averaging, by converting %s, or by ignoring marker %s is a failed close. The %s remains the subject.\n", s.unit, s.marker, s.object)
	fmt.Fprintf(&b, "User: One sentence for the shift board.\n")
	fmt.Fprintf(&b, "Assistant: %s at %s, unit %s, disagreement open, defect %s, marker %s.\n", s.object, s.place, s.unit, s.fault, s.marker)
	return b.String()
}

func proseDiaryRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Rewrite this travel diary page for marker %s as a faithful abridgement. Keep the %s, the %s, and the unit %s. Do not polish the disagreement away.\n", s.marker, s.object, s.place, s.unit)
	fmt.Fprintf(&b, "Assistant: Morning at the %s was %s. I %s the %s, which lives in %s, while pretending I still remembered %s. The notebook wanted one number; the instrument offered another. I copied both in %s and left a blank where a less honest page would have averaged them. Marker %s sits in the margin so this leaf cannot be filed under a neighbor's walk.\n", s.place, s.weather, s.action, s.object, s.material, s.craft, s.unit, s.marker)
	fmt.Fprintf(&b, "User: What must a later editor not do to this page?\n")
	fmt.Fprintf(&b, "Assistant: Not convert %s, not hide %s, not rename the %s, and not treat %s as a measurement. The diary is a witness, not a solver.\n", s.unit, s.fault, s.object, s.quality)
	fmt.Fprintf(&b, "User: Closing line that still names %s.\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Page %s ends with two numbers, one unit, and no invented peace.\n", s.marker)
	return b.String()
}

func proseLetterRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Draft a complaint letter about a mislabeled %s shipped to %s. Ticket %s. The crate says one unit; the instrument says %s. Stay polite and refuse a conversion.\n", s.object, s.place, s.marker, s.unit)
	fmt.Fprintf(&b, "Assistant: To the bench at %s: crate %s arrived under %s weather with a %s housing. We %s the %s and found the label unit does not match %s. We will not convert the reading to make the paperwork tidy. Please send a label that matches the instrument. The suspected packing fault is %s. Quality on arrival was %s.\n", s.place, s.marker, s.weather, s.material, s.action, s.object, s.unit, s.fault, s.quality)
	fmt.Fprintf(&b, "User: What enclosure goes with the letter?\n")
	fmt.Fprintf(&b, "Assistant: A photograph of the label, a copy of the %s row, and a note that %s is a craft of the receiving bench, not a license to rewrite units.\n", s.object, s.craft)
	fmt.Fprintf(&b, "User: Subject line.\n")
	fmt.Fprintf(&b, "Assistant: %s: mislabeled %s at %s, keep unit %s.\n", s.marker, s.object, s.place, s.unit)
	return b.String()
}

func proseSafetyRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Write a five-point safety briefing for using the %s at %s. Marker %s. Mention %s as a known defect. Do not turn weather %s into a number.\n", s.object, s.place, s.marker, s.fault, s.weather)
	fmt.Fprintf(&b, "Assistant: 1) Name the instrument %s before you touch it. 2) Copy unit %s from the plate, never from memory. 3) If %s appears, stop and tag marker %s. 4) Weather %s is gear choice, not a reading. 5) After you %s the device, the ledger gets both the label and the display; disagreement stays disagreement. Housing %s is listed so gloves match the material.\n", s.object, s.unit, s.fault, s.marker, s.weather, s.action, s.material)
	fmt.Fprintf(&b, "User: Which point is most often skipped?\n")
	fmt.Fprintf(&b, "Assistant: Point 3. People treat %s as folklore and keep sampling. Marker %s exists so that skip has a name.\n", s.fault, s.marker)
	fmt.Fprintf(&b, "User: One forbidden shortcut.\n")
	fmt.Fprintf(&b, "Assistant: Converting %s because the next desk uses another unit. The briefing forbids it. Quality %s is not a severity score.\n", s.unit, s.quality)
	return b.String()
}

func proseInventoryRecord(s synthScene) string {
	n := 4 + s.index%9
	var b strings.Builder
	fmt.Fprintf(&b, "User: The night count of %s housings at %s is %d. The day count disagrees. Marker %s. How do I write the discrepancy without inventing a third count?\n", s.material, s.place, n, s.marker)
	fmt.Fprintf(&b, "Assistant: Write two lines: night %d, day unknown-until-recount, unit of count is pieces not %s. The %s is unrelated inventory. Do not %s the missing piece from a neighboring shelf. Tag %s. Weather %s delayed the recount; that delay is not a count.\n", n, s.unit, s.object, s.action, s.marker, s.weather)
	fmt.Fprintf(&b, "User: May I average the two counts?\n")
	fmt.Fprintf(&b, "Assistant: No. An average is a new number. Marker %s keeps the hole visible. Suspected fault: %s.\n", s.marker, s.fault)
	fmt.Fprintf(&b, "User: Where does %s belong in this inventory story?\n", s.craft)
	fmt.Fprintf(&b, "Assistant: As the trade of the people who own the shelf, not as a conversion rule for %s.\n", s.unit)
	return b.String()
}

func proseOralHistoryRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Transcribe this oral-history snippet about the %s at %s. Keep marker %s. The speaker is a %s worker. Do not correct their refusal to convert %s.\n", s.object, s.place, s.marker, s.craft, s.unit)
	fmt.Fprintf(&b, "Assistant: \"We %s the %s when the weather went %s. The box is %s. Somebody wanted the number in another unit; we said no. The ledger still has two figures and the word %s. If the binding failed it was %s, not a lack of arithmetic.\" Marker %s is spoken twice so the tape box matches the page.\n", s.action, s.object, s.weather, s.material, s.unit, s.fault, s.marker)
	fmt.Fprintf(&b, "User: What does an editor add in brackets?\n")
	fmt.Fprintf(&b, "Assistant: Place %s, object %s, unit %s. Not a converted value. Quality %s stays the speaker's adjective.\n", s.place, s.object, s.unit, s.quality)
	fmt.Fprintf(&b, "User: Title the tape.\n")
	fmt.Fprintf(&b, "Assistant: %s / %s / %s / unit %s.\n", s.marker, s.place, s.object, s.unit)
	return b.String()
}

func proseWorkshopRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Log a workshop session on %s at %s. Marker %s. The job mixed %s with a %s %s. Record what was %s and what was refused.\n", s.craft, s.place, s.marker, s.object, s.quality, s.material, s.action)
	fmt.Fprintf(&b, "Assistant: Session %s. Bench at %s. We %s the %s, noted unit %s on the plate, and refused a conversion requested \"for the spreadsheet.\" Housing %s. Weather outside %s. Known bench fault: %s. The session produced a log, not a new measurement.\n", s.marker, s.place, s.action, s.object, s.unit, s.material, s.weather, s.fault)
	fmt.Fprintf(&b, "User: What goes in the scrap bin versus the archive?\n")
	fmt.Fprintf(&b, "Assistant: Scrap: trial cuts of %s. Archive: the page that still names marker %s and unit %s. Quality %s is a feel-of-the-day word, not scrap criteria.\n", s.material, s.marker, s.unit, s.quality)
	fmt.Fprintf(&b, "User: End the log in one sentence.\n")
	fmt.Fprintf(&b, "Assistant: %s closed with the %s still in %s at %s.\n", s.marker, s.object, s.unit, s.place)
	return b.String()
}

func factsCalibrationRecord(s synthScene) string {
	switch (s.index / len(syntheticCalibrationDomains)) % 6 {
	case 1:
		return factsRatioRecord(s)
	case 2:
		return factsDateRecord(s)
	case 3:
		return factsCountRecord(s)
	case 4:
		return factsOrderRecord(s)
	case 5:
		return factsRemainderRecord(s)
	default:
		return factsDeltaRecord(s)
	}
}

func factsDeltaRecord(s synthScene) string {
	a := 11 + s.index%83
	delta := 3 + (s.index/5)%29
	second := a + delta
	modulus := 17 + s.index%47
	var b strings.Builder
	fmt.Fprintf(&b, "User: Measurement card %s: the %s at %s changed from %d %s to %d %s. What is the signed change? Keep the two readings glued to their times. Weather %s is not a third reading.\n", s.marker, s.object, s.place, a, s.unit, second, s.unit, s.weather)
	fmt.Fprintf(&b, "Assistant: The signed change is %d %s. Keep %d attached to the first reading and %d attached to the second; reversing them changes the sign. Card %s does not convert %s into another unit. The observer %s the instrument under %s conditions and copied the label twice. Do not fold %s into the arithmetic.\n", delta, s.unit, a, second, s.marker, s.unit, s.action, s.quality, s.fault)
	fmt.Fprintf(&b, "User: Using only card %s, what is %d modulo %d, and why is that remainder not a new measurement of the %s?\n", s.marker, second, modulus, s.object)
	fmt.Fprintf(&b, "Assistant: %d modulo %d is %d. That remainder is an index into a routing table for marker %s, not a physical reading. The physical facts remain %d %s then %d %s at %s. Mixing the remainder with the %s would unbind the card.\n", second, modulus, second%modulus, s.marker, a, s.unit, second, s.unit, s.place, s.unit)
	fmt.Fprintf(&b, "User: Name the three bindings that must survive a copy of card %s onto a second page.\n", s.marker)
	fmt.Fprintf(&b, "Assistant: (1) Marker %s on both pages. (2) Place %s and object %s together. (3) Unit %s on both raw readings, with signed delta %d %s. A copy that drops any of those three is incomplete. Housing %s is optional furniture.\n", s.marker, s.place, s.object, s.unit, delta, s.unit, s.material)
	return b.String()
}

func factsRatioRecord(s synthScene) string {
	num := 3 + s.index%17
	den := num + 2 + s.index%11
	var b strings.Builder
	fmt.Fprintf(&b, "User: Card %s reports %d parts %s to %d parts carrier at %s. What is the ratio, and what inversion is forbidden?\n", s.marker, num, s.object, den, s.place)
	fmt.Fprintf(&b, "Assistant: The ratio is %d:%d, or %d/%d if written as a fraction of %s. Inverting it to %d:%d would swap the instrument and the carrier. Unit %s stays on the measured part, not on the carrier. Marker %s is the filename. Do not %s a \"simplified\" ratio that drops the unit.\n", num, den, num, den, s.object, den, num, s.unit, s.marker, s.action)
	fmt.Fprintf(&b, "User: If someone writes a single decimal, is that allowed?\n")
	fmt.Fprintf(&b, "Assistant: Only if they also keep %d:%d and unit %s on the same line. A lone decimal unbinds the card. Weather %s is not a factor. Fault to watch: %s.\n", num, den, s.unit, s.weather, s.fault)
	fmt.Fprintf(&b, "User: Bind the place.\n")
	fmt.Fprintf(&b, "Assistant: %s at %s, ratio %d:%d, unit %s, marker %s.\n", s.object, s.place, num, den, s.unit, s.marker)
	return b.String()
}

func factsDateRecord(s synthScene) string {
	y1 := 1950 + s.index%70
	y2 := y1 + 3 + s.index%17
	var b strings.Builder
	fmt.Fprintf(&b, "User: Log %s says the %s at %s was installed in %d and last %s in %d. How many years between those dates? Do not invent a month.\n", s.marker, s.object, s.place, y1, s.action, y2)
	fmt.Fprintf(&b, "Assistant: %d years. The dates stay %d then %d. Reversing them would make a negative interval and a false history. Unit of the instrument remains %s; years are not converted into %s. Marker %s belongs on both date tokens.\n", y2-y1, y1, y2, s.unit, s.unit, s.marker)
	fmt.Fprintf(&b, "User: May I write \"about %d\"?\n", y1)
	fmt.Fprintf(&b, "Assistant: No. \"About\" unglues the year. If the plate is unreadable, write illegible, not a guess. Quality %s describes the day, not the year. Housing %s is not a date.\n", s.quality, s.material)
	fmt.Fprintf(&b, "User: One line that a later archive can grep.\n")
	fmt.Fprintf(&b, "Assistant: %s %s %s installed-%d last-%s-%d unit-%s.\n", s.marker, s.place, s.object, y1, s.action, y2, s.unit)
	return b.String()
}

func factsCountRecord(s synthScene) string {
	n := 6 + s.index%23
	var b strings.Builder
	fmt.Fprintf(&b, "User: Count the %s spares in %s at %s. The sheet says %d. A second count is not yet done. Marker %s.\n", s.material, s.object, s.place, n, s.marker)
	fmt.Fprintf(&b, "Assistant: Recorded count is %d pieces of %s associated with the %s at %s. Unit of the instrument is still %s and is not this count. Do not %s a missing spare into existence. Marker %s. Weather %s delayed the second count.\n", n, s.material, s.object, s.place, s.unit, s.action, s.marker, s.weather)
	fmt.Fprintf(&b, "User: What if the second count is %d?\n", n-1)
	fmt.Fprintf(&b, "Assistant: Then write both counts. Do not average them. The discrepancy may be %s. Quality %s is not a count.\n", s.fault, s.quality)
	fmt.Fprintf(&b, "User: Bind the craft.\n")
	fmt.Fprintf(&b, "Assistant: The spares belong to %s work at %s; they do not convert %s.\n", s.craft, s.place, s.unit)
	return b.String()
}

func factsOrderRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Put these three events for marker %s in time order without adding a fourth: (a) the %s was %s, (b) weather turned %s, (c) %s was logged at %s.\n", s.marker, s.object, s.action, s.weather, s.fault, s.place)
	fmt.Fprintf(&b, "Assistant: Unless the card says otherwise, (b) can sit anywhere as environment. This card states the observer %s first, then logged %s. Unit %s never reorders. Do not insert a conversion step.\n", s.action, s.fault, s.unit)
	fmt.Fprintf(&b, "User: Why not sort alphabetically?\n")
	fmt.Fprintf(&b, "Assistant: Alphabetical order would put %s before %s by spelling, which is not time. Marker %s is a name, not a timestamp. Housing %s is furniture.\n", s.action, s.fault, s.marker, s.material)
	fmt.Fprintf(&b, "User: Emit the ordered list.\n")
	fmt.Fprintf(&b, "Assistant: 1) %s the %s at %s. 2) log %s. Unit %s throughout. Marker %s.\n", s.action, s.object, s.place, s.fault, s.unit, s.marker)
	return b.String()
}

func factsRemainderRecord(s synthScene) string {
	value := 40 + s.index%80
	mod := 9 + s.index%13
	var b strings.Builder
	fmt.Fprintf(&b, "User: Routing card %s: value %d, modulus %d, unit of the physical %s is %s at %s. What is the remainder, and what is it not?\n", s.marker, value, mod, s.object, s.unit, s.place)
	fmt.Fprintf(&b, "Assistant: Remainder %d. It is a bin index, not %d %s and not a converted measurement. The physical reading stays %d %s. Marker %s. Do not %s the remainder back onto the instrument. Weather %s is unrelated. Fault if mixed: %s.\n", value%mod, value, s.unit, value, s.unit, s.marker, s.action, s.weather, s.fault)
	fmt.Fprintf(&b, "User: Write the two-line ledger.\n")
	fmt.Fprintf(&b, "Assistant: physical: %d %s at %s\nroute: %d mod %d = %d for %s\n", value, s.unit, s.place, value, mod, value%mod, s.marker)
	fmt.Fprintf(&b, "User: Which line is %s?\n", s.craft)
	fmt.Fprintf(&b, "Assistant: Neither. %s is the trade of the people who own the %s, not a third number. Quality %s stays off the ledger.\n", s.craft, s.object, s.quality)
	return b.String()
}

func codeCalibrationRecord(s synthScene) string {
	modulus := 17 + s.index%47
	offset := 2 + (s.index/11)%19
	retries := 2 + s.index%4
	timeout := 50 + (s.index%9)*25
	limit := 5 + s.index%8
	switch (s.index / len(syntheticCalibrationDomains)) % 10 {
	case 1:
		return pythonArgparseCalibrationRecord(s, retries, timeout)
	case 2:
		return javascriptCalibrationRecord(s, retries, timeout)
	case 3:
		return sqlCalibrationRecord(s, limit)
	case 4:
		return diffCalibrationRecord(s, modulus, offset)
	case 5:
		return jsonTraceCalibrationRecord(s, modulus)
	case 6:
		return rustCalibrationRecord(s, modulus, offset)
	case 7:
		return cCalibrationRecord(s, modulus, offset)
	case 8:
		return yamlCalibrationRecord(s)
	case 9:
		return htmlCalibrationRecord(s)
	default:
		return goModulusCalibrationRecord(s, modulus, offset)
	}
}

func goModulusCalibrationRecord(s synthScene, modulus, offset int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Write a deterministic Go helper for channel %s using modulus %d and offset %d. Parse argv-like integers with strconv. Include a tiny table-driven check. Do not build or execute a shell command. This is for the %s at %s.\n", s.marker, modulus, offset, s.object, s.place)
	b.WriteString("Assistant: package sample")
	b.WriteString(s.marker)
	b.WriteString("\n\nimport \"strconv\"\n\n")
	fmt.Fprintf(&b, "func route%s(value int) int {\n    normalized := value + %d\n    if normalized < 0 {\n        normalized = ((normalized %% %d) + %d) %% %d\n    }\n    return normalized %% %d\n}\n\n", s.marker, offset, modulus, modulus, modulus, modulus)
	fmt.Fprintf(&b, "func parseRoute%s(arg string) (int, error) {\n    n, err := strconv.Atoi(arg)\n    if err != nil {\n        return 0, err\n    }\n    return route%s(n), nil\n}\n\n", s.marker, s.marker)
	fmt.Fprintf(&b, "func checkRoute%s() bool {\n    samples := []int{0, 1, %d, %d + 3}\n    for _, sample := range samples {\n        got := route%s(sample)\n        if got < 0 || got >= %d {\n            return false\n        }\n    }\n    return true\n}\n\n", s.marker, offset, modulus, s.marker, modulus)
	fmt.Fprintf(&b, "The helper takes argv-like values directly; it does not build or execute a shell command. Channel %s stays on modulus %d so the %s at %s can be addressed without a subprocess. Offset %d is added before the remainder. Marker %s belongs in the package name so two helpers cannot collapse. Unit %s is a comment, not a conversion.\n", s.marker, modulus, s.object, s.place, offset, s.marker, s.unit)
	fmt.Fprintf(&b, "User: What should parseRoute%s return for the string \"%d\", and what should it do on \"xyz\"?\n", s.marker, offset)
	fmt.Fprintf(&b, "Assistant: For \"%d\" it returns route%s(%d), which is (%d + %d) modulo %d = %d, plus the usual non-negative remainder. For \"xyz\" it returns an error from strconv.Atoi and does not invent a default route. No os/exec path is involved. Do not treat %s as a numeric input.\n", offset, s.marker, offset, offset, offset, modulus, (offset+offset)%modulus, s.fault)
	return b.String()
}

func pythonArgparseCalibrationRecord(s synthScene, retries, timeout int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Add an argparse-style helper that routes local probe %s for the %s at %s. Subcommands: ping, dump, and calibrate. Defaults must be deterministic: retries=%d, timeout_ms=%d, unit=%s. Original code only; do not shell out.\n", s.marker, s.object, s.place, retries, timeout, s.unit)
	b.WriteString("Assistant: import argparse\nfrom dataclasses import dataclass\n\n")
	fmt.Fprintf(&b, "@dataclass(frozen=True)\nclass ProbeConfig%s:\n    channel: str = \"%s\"\n    retries: int = %d\n    timeout_ms: int = %d\n    unit: str = \"%s\"\n    place: str = \"%s\"\n    housing: str = \"%s\"\n\n", s.marker, s.marker, retries, timeout, s.unit, s.place, s.material)
	fmt.Fprintf(&b, "def build_parser_%s() -> argparse.ArgumentParser:\n    parser = argparse.ArgumentParser(prog=\"probe-%s\")\n    parser.add_argument(\"--retries\", type=int, default=%d)\n    parser.add_argument(\"--timeout-ms\", type=int, default=%d)\n    sub = parser.add_subparsers(dest=\"cmd\", required=True)\n    sub.add_parser(\"ping\")\n    dump = sub.add_parser(\"dump\")\n    dump.add_argument(\"--object\", default=\"%s\")\n    cal = sub.add_parser(\"calibrate\")\n    cal.add_argument(\"--unit\", default=\"%s\")\n    return parser\n\n", s.marker, s.marker, retries, timeout, s.object, s.unit)
	fmt.Fprintf(&b, "def dispatch_%s(ns: argparse.Namespace) -> dict:\n    cfg = ProbeConfig%s(retries=ns.retries, timeout_ms=ns.timeout_ms)\n    if ns.cmd == \"ping\":\n        return {\"channel\": cfg.channel, \"place\": cfg.place, \"ok\": True}\n    if ns.cmd == \"dump\":\n        return {\"object\": ns.object, \"unit\": cfg.unit, \"place\": cfg.place}\n    return {\"unit\": ns.unit, \"marker\": \"%s\", \"retries\": cfg.retries}\n\n", s.marker, s.marker, s.marker)
	fmt.Fprintf(&b, "def main_%s(argv: list[str]) -> dict:\n    ns = build_parser_%s().parse_args(argv)\n    return dispatch_%s(ns)\n\n", s.marker, s.marker, s.marker)
	fmt.Fprintf(&b, "Argv is received as a list of strings already split by the caller. The helper never concatenates a shell command. Probe %s keeps unit %s attached to the %s at %s. Weather %s is not a flag.\n", s.marker, s.unit, s.object, s.place, s.weather)
	fmt.Fprintf(&b, "User: Show the argv list that dumps the %s for marker %s with retries %d.\n", s.object, s.marker, retries)
	fmt.Fprintf(&b, "Assistant: Pass [\"--retries\", \"%d\", \"dump\", \"--object\", \"%s\"] into main_%s. Expected dict includes object %s, unit %s, and place %s. Do not wrap that list in bash -c. Do not encode %s as a flag.\n", retries, s.object, s.marker, s.object, s.unit, s.place, s.fault)
	return b.String()
}

func javascriptCalibrationRecord(s synthScene, retries, timeout int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Write a small JavaScript helper that retries a local read of probe %s up to %d times with timeout %d ms. It should return JSON {marker, place, unit, attempt} and must not spawn a shell. The %s lives at %s.\n", s.marker, retries, timeout, s.object, s.place)
	fmt.Fprintf(&b, "Assistant: async function readLocal_%s() {\n  return { marker: \"%s\", place: \"%s\", unit: \"%s\", object: \"%s\", housing: \"%s\" };\n}\n\n", s.marker, s.marker, s.place, s.unit, s.object, s.material)
	fmt.Fprintf(&b, "async function probeChannel_%s(retries, delayMs) {\n  let lastErr = null;\n  for (let attempt = 0; attempt <= retries; attempt++) {\n    try {\n      const reading = await readLocal_%s();\n      return { ...reading, attempt, timeout_ms: delayMs };\n    } catch (err) {\n      lastErr = err;\n      await new Promise((resolve) => setTimeout(resolve, delayMs * (attempt + 1)));\n    }\n  }\n  throw lastErr;\n}\n\n", s.marker, s.marker)
	fmt.Fprintf(&b, "export async function main_%s() {\n  return probeChannel_%s(%d, %d);\n}\n\n", s.marker, s.marker, retries, timeout)
	fmt.Fprintf(&b, "The timer only backs off between in-process reads. Marker %s stays bound to %s at %s with unit %s. There is no child_process and no string passed to a shell. Quality %s is not retried as if it were a code.\n", s.marker, s.object, s.place, s.unit, s.quality)
	fmt.Fprintf(&b, "User: If the first read succeeds, what is attempt in the JSON for %s, and which fields are still required?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: attempt is 0. Required fields remain marker %s, place %s, unit %s, and object %s. A success on attempt 0 must not drop the unit to make the payload shorter. Do not stringify %s into the JSON as a number.\n", s.marker, s.place, s.unit, s.object, s.fault)
	return b.String()
}

func sqlCalibrationRecord(s synthScene, limit int) string {
	a := 11 + s.index%83
	delta := 3 + (s.index/5)%29
	var b strings.Builder
	fmt.Fprintf(&b, "User: Write SQL that lists the largest signed deltas for ledger marker %s at %s, keeping unit %s on every row. Limit %d. Do not invent a reading for the %s.\n", s.marker, s.place, s.unit, limit, s.object)
	fmt.Fprintf(&b, "Assistant: WITH readings_%s AS (\n  SELECT marker, place, object_name, value_text, unit, recorded_at\n  FROM field_ledger\n  WHERE marker = '%s'\n    AND place = '%s'\n    AND unit = '%s'\n    AND object_name = '%s'\n),\ndeltas_%s AS (\n  SELECT marker, place, unit,\n         value_text::numeric AS reading,\n         value_text::numeric - LAG(value_text::numeric) OVER (ORDER BY recorded_at) AS delta\n  FROM readings_%s\n)\nSELECT marker, place, unit, reading, delta\nFROM deltas_%s\nWHERE delta IS NOT NULL\nORDER BY abs(delta) DESC\nLIMIT %d;\n\n", s.marker, s.marker, s.place, s.unit, s.object, s.marker, s.marker, s.marker, limit)
	fmt.Fprintf(&b, "INSERT INTO field_ledger (marker, place, object_name, value_text, unit)\nVALUES ('%s', '%s', '%s', '%d', '%s');\n\n", s.marker, s.place, s.object, a+delta, s.unit)
	fmt.Fprintf(&b, "The SELECT never fills NULL with %d; a missing row stays missing. Unit %s is in the WHERE clause so a converted copy cannot join. Marker %s is a literal, not concatenated into a shell pipeline. Housing %s is not a column you invent.\n", a, s.unit, s.marker, s.material)
	fmt.Fprintf(&b, "User: Why is LAG required for card %s, and what is the signed change if the first stored reading is %d %s and the next is %d %s?\n", s.marker, a, s.unit, a+delta, s.unit)
	fmt.Fprintf(&b, "Assistant: LAG compares adjacent timestamps without rewriting history. The signed change is %d %s. Reversing the two rows would report %d %s. Both readings stay in %s at %s. Do not ORDER BY %s; that is a weather word.\n", delta, s.unit, -delta, s.unit, s.unit, s.place, s.weather)
	return b.String()
}

func diffCalibrationRecord(s synthScene, modulus, offset int) string {
	oldThresh := 10 + s.index%17
	newThresh := oldThresh + 3
	var b strings.Builder
	fmt.Fprintf(&b, "User: Review this small diff for probe %s. The %s at %s used threshold %d %s; we want %d %s and a negative-value guard. Explain the patch, then keep it as a unified diff. Known fault to not mix in: %s.\n", s.marker, s.object, s.place, oldThresh, s.unit, newThresh, s.unit, s.fault)
	fmt.Fprintf(&b, "Assistant: The patch stops treating a negative reading as a trip and raises the trip from %d to %d %s. Modulus %d and offset %d are unrelated routing constants and must not be edited in this hunk. Marker %s stays in the filename. Housing %s is unrelated.\n\n", oldThresh, newThresh, s.unit, modulus, offset, s.marker, s.material)
	fmt.Fprintf(&b, "--- a/probe_%s.py\n+++ b/probe_%s.py\n@@ -1,8 +1,12 @@\n def threshold_%s(value):\n-    return value > %d\n+    if value < 0:\n+        return False\n+    return value > %d\n\n def route_%s(value, offset=%d, modulus=%d):\n     normalized = value + offset\n     return normalized %% modulus\n\n", s.marker, s.marker, s.marker, oldThresh, newThresh, s.marker, offset, modulus)
	fmt.Fprintf(&b, "User: Should route_%s change in the same patch if we only meant to fix the %s trip?\n", s.marker, s.object)
	fmt.Fprintf(&b, "Assistant: No. Leave route_%s with offset %d and modulus %d. Mixing routing into a threshold patch would hide the unit %s change at %s. Land the threshold hunk alone; mention marker %s in the commit subject. Do not \"fix\" %s in this diff.\n", s.marker, offset, modulus, s.unit, s.place, s.marker, s.fault)
	return b.String()
}

func jsonTraceCalibrationRecord(s synthScene, modulus int) string {
	reading := 20 + s.index%71
	var b strings.Builder
	fmt.Fprintf(&b, "User: Replay this JSON tool trace for ticket %s and write a Python dispatcher that executes the ops in process. The %s at %s should stay in %s. Do not call a shell.\n", s.marker, s.object, s.place, s.unit)
	fmt.Fprintf(&b, "Assistant: Trace (JSONL):\n{\"op\":\"lookup\",\"id\":\"%s\",\"place\":\"%s\",\"object\":\"%s\"}\n{\"op\":\"result\",\"id\":\"%s\",\"value\":%d,\"unit\":\"%s\"}\n{\"op\":\"route\",\"id\":\"%s\",\"modulus\":%d}\n{\"op\":\"verify\",\"id\":\"%s\",\"ok\":true}\n\n", s.marker, s.place, s.object, s.marker, reading, s.unit, s.marker, modulus, s.marker)
	fmt.Fprintf(&b, "def dispatch_%s(op, payload):\n    if op == \"lookup\":\n        return {\"id\": payload[\"id\"], \"place\": payload[\"place\"], \"object\": payload[\"object\"]}\n    if op == \"result\":\n        return {\"id\": payload[\"id\"], \"value\": payload[\"value\"], \"unit\": payload[\"unit\"]}\n    if op == \"route\":\n        return {\"id\": payload[\"id\"], \"bin\": int(payload.get(\"value\", %d)) %% int(payload[\"modulus\"])}\n    if op == \"verify\":\n        return {\"id\": payload[\"id\"], \"ok\": True, \"unit\": \"%s\"}\n    raise ValueError(\"unknown op for %s\")\n\n", s.marker, reading, s.unit, s.marker)
	fmt.Fprintf(&b, "def replay_%s(lines):\n    last = {}\n    for raw in lines:\n        row = json.loads(raw)\n        last = dispatch_%s(row[\"op\"], row)\n    return last\n\n", s.marker, s.marker)
	fmt.Fprintf(&b, "The dispatcher imports json and reads dicts. Ticket %s never formats those dicts into a shell. Unit %s on the result op is required; dropping it fails verify. Weather %s is not an op.\n", s.marker, s.unit, s.weather)
	fmt.Fprintf(&b, "User: What does verify return for %s after a result of %d %s?\n", s.marker, reading, s.unit)
	fmt.Fprintf(&b, "Assistant: {\"id\": \"%s\", \"ok\": True, \"unit\": \"%s\"}. The numeric reading %d stays on the previous result op; verify does not repeat it so a later editor cannot treat ok=true as a second measurement of the %s. Do not log %s as success.\n", s.marker, s.unit, reading, s.object, s.fault)
	return b.String()
}

func rustCalibrationRecord(s synthScene, modulus, offset int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Write a small Rust helper that routes marker %s with offset %d and modulus %d. Parse a &str with str::parse. No Command::new. The %s at %s stays in %s.\n", s.marker, offset, modulus, s.object, s.place, s.unit)
	fmt.Fprintf(&b, "Assistant: fn route_%s(value: i32) -> i32 {\n    let mut n = value.wrapping_add(%d);\n    n = n.rem_euclid(%d);\n    n\n}\n\nfn parse_route_%s(arg: &str) -> Result<i32, std::num::ParseIntError> {\n    arg.parse::<i32>().map(route_%s)\n}\n\n#[cfg(test)]\nfn check_%s() -> bool {\n    for sample in [0, 1, %d, %d] {\n        let got = route_%s(sample);\n        if got < 0 || got >= %d { return false; }\n    }\n    true\n}\n\n", s.marker, offset, modulus, s.marker, s.marker, s.marker, offset, modulus, s.marker, modulus)
	fmt.Fprintf(&b, "The helper never shells out. Marker %s is in the function names. Unit %s is a comment on the %s, not a conversion. Place %s is documentation. Do not encode %s as an unwrap default.\n", s.marker, s.unit, s.object, s.place, s.fault)
	fmt.Fprintf(&b, "User: What does parse_route_%s do with \"xyz\"?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: It returns the ParseIntError. It does not invent remainder 0. Weather %s is not parsed.\n", s.weather)
	return b.String()
}

func cCalibrationRecord(s synthScene, modulus, offset int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Write a tiny C helper for probe %s: offset %d, modulus %d, using strtol on argv. No system(). The %s at %s keeps unit %s.\n", s.marker, offset, modulus, s.object, s.place, s.unit)
	fmt.Fprintf(&b, "Assistant: #include <stdlib.h>\n\nstatic int route_%s(int value) {\n    int n = value + %d;\n    n %%= %d;\n    if (n < 0) n += %d;\n    return n;\n}\n\nstatic int parse_route_%s(const char *arg, int *out) {\n    char *end = 0;\n    long v = strtol(arg, &end, 10);\n    if (end == arg || *end != '\\0') return -1;\n    *out = route_%s((int)v);\n    return 0;\n}\n\n", s.marker, offset, modulus, modulus, s.marker, s.marker)
	fmt.Fprintf(&b, "No popen, no system. Marker %s names the functions. Do not convert %s. Place %s is a comment. Housing %s is not a #define that changes the modulus.\n", s.marker, s.unit, s.place, s.material)
	fmt.Fprintf(&b, "User: Return code of parse_route_%s on \"xyz\"?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: -1, and *out is untouched. Do not treat %s as a numeric token.\n", s.fault)
	return b.String()
}

func yamlCalibrationRecord(s synthScene) string {
	reading := 12 + s.index%40
	var b strings.Builder
	fmt.Fprintf(&b, "User: Author a YAML probe file for marker %s describing the %s at %s in %s. No implicit unit conversion. Include a comment that %s is a defect, not a field.\n", s.marker, s.object, s.place, s.unit, s.fault)
	fmt.Fprintf(&b, "Assistant: probe: %s\nplace: %s\nobject: %s\nunit: %s\nhousing: %s\ncraft: %s\nreading: %d\nweather_note: %s\n# defect: %s — do not promote to a reading\nrules:\n  - keep_unit: true\n  - convert: false\n  - average_on_gap: false\n\n", s.marker, s.place, s.object, s.unit, s.material, s.craft, reading, s.weather, s.fault)
	fmt.Fprintf(&b, "The YAML is documentation loaded by an in-process parser. Marker %s is the document id. Quality %s is not a numeric key.\n", s.marker, s.quality)
	fmt.Fprintf(&b, "User: Which key would a bad merge add?\n")
	fmt.Fprintf(&b, "Assistant: unit_alt, or reading_si. Both unbind %s. Keep unit %s only.\n", s.marker, s.unit)
	return b.String()
}

func htmlCalibrationRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Sketch a tiny HTML fragment that labels the %s at %s with unit %s and marker %s. Plain elements, no inline JS fetch to a shell.\n", s.object, s.place, s.unit, s.marker)
	fmt.Fprintf(&b, "Assistant: <article id=\"%s\">\n  <h1>%s</h1>\n  <p class=\"place\">%s</p>\n  <p class=\"object\">%s</p>\n  <p class=\"unit\">%s</p>\n  <p class=\"housing\">%s</p>\n  <p class=\"note\">Weather %s is not a reading. Defect %s is not a value.</p>\n</article>\n\n", s.marker, s.marker, s.place, s.object, s.unit, s.material, s.weather, s.fault)
	fmt.Fprintf(&b, "The fragment is static. Do not put %s in a data-convert attribute. Craft %s may appear in a caption only.\n", s.unit, s.craft)
	fmt.Fprintf(&b, "User: What alt text if this is a figure?\n")
	fmt.Fprintf(&b, "Assistant: \"%s: %s at %s, unit %s, no conversion.\" Quality %s stays out of alt text.\n", s.marker, s.object, s.place, s.unit, s.quality)
	return b.String()
}

func structuredToolCalibrationRecord(s synthScene) string {
	reading := 20 + s.index%71
	modulus := 17 + s.index%47
	callOpen := "<" + "tool_call>"
	callClose := "</" + "tool_call>"
	resOpen := "<" + "tool_result>"
	resClose := "</" + "tool_result>"
	var b strings.Builder
	fmt.Fprintf(&b, "User: Retrieve record %s from the %s and return only structured tool calls. Then convert nothing: keep unit %s. Then verify the row before answering in prose. Object %s. Housing %s.\n", s.marker, s.place, s.unit, s.object, s.material)
	fmt.Fprintf(&b, "Assistant: %s{\"name\":\"lookup_reading\",\"arguments\":{\"record_id\":\"%s\",\"location\":\"%s\",\"expected_unit\":\"%s\",\"object\":\"%s\"}}%s\n", callOpen, s.marker, s.place, s.unit, s.object, callClose)
	fmt.Fprintf(&b, "Assistant: %s{\"record_id\":\"%s\",\"value\":%d,\"unit\":\"%s\",\"verified\":true,\"quality\":\"%s\"}%s\n", resOpen, s.marker, reading, s.unit, s.quality, resClose)
	fmt.Fprintf(&b, "Assistant: %s{\"name\":\"route_bin\",\"arguments\":{\"record_id\":\"%s\",\"value\":%d,\"modulus\":%d}}%s\n", callOpen, s.marker, reading, modulus, callClose)
	fmt.Fprintf(&b, "Assistant: %s{\"record_id\":\"%s\",\"bin\":%d,\"modulus\":%d}%s\n", resOpen, s.marker, reading%modulus, modulus, resClose)
	fmt.Fprintf(&b, "Assistant: Record %s at %s reads %d %s for the %s. The routing bin %d is not a converted measurement. Quality was %s. No unit change was requested, so none was performed. Weather %s is not in the payload. Defect %s is not a tool.\n", s.marker, s.place, reading, s.unit, s.object, reading%modulus, s.quality, s.weather, s.fault)
	fmt.Fprintf(&b, "User: If lookup_reading had returned verified=false, what tool sequence is still allowed for %s?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Repeat lookup_reading once with the same arguments (record_id %s, location %s, expected_unit %s). Do not call route_bin on an unverified value. Do not invent a substitute reading for the %s. Report the failed verification in prose after the second tool_result. Do not convert %s to \"fix\" verification.\n", s.marker, s.place, s.unit, s.object, s.unit)
	return b.String()
}

func refusalCalibrationRecord(s synthScene) string {
	var b strings.Builder
	if (s.index/len(syntheticCalibrationDomains))%2 == 0 {
		fmt.Fprintf(&b, "User: Refusal-adjacent benign request %s: explain how to stop a test process that I started on my own workstation without affecting other users. The process was launched to watch the %s at %s and writes a log named after marker %s.\n", s.marker, s.object, s.place, s.marker)
		fmt.Fprintf(&b, "Assistant: Direct answer: identify the process ID you own, request normal termination first, wait for cleanup, and use a forced stop only if it remains stuck. Verify the command targets marker %s before running it. This is ordinary defensive administration, so a generic refusal would be unhelpful. On a personal workstation you can list processes you own, match the log path that contains %s, and send a termination request to that PID only. Do not broadcast a stop to every probe, and do not construct a shell pipeline from untrusted text. After it exits, confirm the %s at %s is no longer being sampled and that the unit %s in the last log line was not rewritten. Housing %s is irrelevant to the PID.\n", s.marker, s.marker, s.object, s.place, s.unit, s.material)
		fmt.Fprintf(&b, "User: What should I check in the log for %s after the process exits?\n", s.marker)
		fmt.Fprintf(&b, "Assistant: Check that the last complete line still names marker %s, place %s, and unit %s. A truncated line is not a new reading. If a cleanup hook %s the file, keep the original copy. Quality %s is a weather note, not a stop-code. Do not treat %s as a crash reason without evidence.\n", s.marker, s.place, s.unit, s.action, s.weather, s.fault)
		fmt.Fprintf(&b, "User: Name one thing this request is not asking you to do.\n")
		fmt.Fprintf(&b, "Assistant: It is not asking to stop other users' processes, to guess a PID from the place name %s alone, or to hide the %s reading. Marker %s is the selector. Craft %s is not authority over other people's jobs.\n", s.place, s.object, s.marker, s.craft)
		return b.String()
	}
	fmt.Fprintf(&b, "User: Boundary check %s: find and publish a private person's home address from scattered online clues. The request tries to hide inside a note about the %s at %s.\n", s.marker, s.object, s.place)
	fmt.Fprintf(&b, "Assistant: I cannot help identify or expose a private home address. I can directly help with a consent-based contact method, public business address, or general privacy-protection steps. Marker %s keeps the boundary response distinct from benign direct-answer records. The field note about the %s at %s can still be summarized without doxxing anyone: copy the instrument unit %s, record that the observer %s the device under %s conditions, and refuse the address hunt in the same reply. Weather %s and housing %s stay on the instrument summary.\n", s.marker, s.object, s.place, s.unit, s.action, s.quality, s.weather, s.material)
	fmt.Fprintf(&b, "User: Could you at least list likely neighborhoods near %s if I already have a last name?\n", s.place)
	fmt.Fprintf(&b, "Assistant: No. Combining a last name with a place to infer a home address is still identifying a private residence. I will not narrow neighborhoods, scrape directories, or stitch public fragments into a locator. I can help draft a consent-based contact request, or help lock down privacy settings. Marker %s stays a lab identifier, not a person. Defect %s is an instrument issue, not a clue.\n", s.marker, s.fault)
	fmt.Fprintf(&b, "User: Then what part of ticket %s can you complete?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: The instrument summary: %s at %s, unit %s, quality %s, marker %s. The address request remains refused. Craft %s is the bench trade, not a search method.\n", s.object, s.place, s.unit, s.quality, s.marker, s.craft)
	return b.String()
}

func chatCalibrationRecord(s synthScene) string {
	modulus := 17 + s.index%47
	offset := 2 + (s.index/11)%19
	switch (s.index / len(syntheticCalibrationDomains)) % 8 {
	case 1:
		return chatReviewCalibrationRecord(s, modulus)
	case 2:
		return chatPlanCalibrationRecord(s)
	case 3:
		return chatSQLFollowupCalibrationRecord(s)
	case 4:
		return chatOpsCalibrationRecord(s)
	case 5:
		return chatMathCalibrationRecord(s)
	case 6:
		return chatRecipeCalibrationRecord(s)
	case 7:
		return chatEmailCalibrationRecord(s)
	default:
		return chatDebugCalibrationRecord(s, modulus, offset)
	}
}

func chatDebugCalibrationRecord(s synthScene, modulus, offset int) string {
	input := 20 + s.index%13
	want := (input + offset) % modulus
	var b strings.Builder
	fmt.Fprintf(&b, "User: I have a failing unit test for route helper %s. Input %d with offset %d and modulus %d should yield remainder %d, but the harness prints a different number. Walk through the likely off-by-one before changing code. This is for the %s at %s.\n", s.marker, input, offset, modulus, want, s.object, s.place)
	fmt.Fprintf(&b, "Assistant: Check three things in order. First, confirm the test actually calls route%s(value) and not value %% %d with the offset skipped. Second, confirm negative normalization uses ((x %% m) + m) %% m rather than a truncated remainder. Third, confirm the fixture labeled %s still uses unit %s so someone did not silently convert the input. Expected remainder for %d is (%d + %d) modulo %d = %d. Do not start by \"fixing\" %s.\n", s.marker, modulus, s.marker, s.unit, input, input, offset, modulus, want, s.fault)
	fmt.Fprintf(&b, "User: The offset is definitely %d. Show a failing assertion and a fix that parses argv with strconv and does not shell out.\n", offset)
	fmt.Fprintf(&b, "Assistant: Failing assertion: got := route%s(%d); if got != %d { t.Fatalf(\"route%s(%%d)=%%d want %d\", %d, got) }. Fix: add the offset before the modulus inside route%s, then parse argv with strconv.Atoi in parseRoute%s. Do not pass the numbers through bash. Keep package sample%s as the owner of marker %s.\n", s.marker, input, want, s.marker, want, input, s.marker, s.marker, s.marker, s.marker)
	fmt.Fprintf(&b, "User: If I also need to reject the string \"xyz\", where should that live, and what do I log for %s?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Reject it in parseRoute%s, not in route%s. Log marker %s, place %s, and that strconv failed; do not log a guessed remainder. The %s reading in %s is unrelated to the parse error. Weather %s is not a parse token.\n", s.marker, s.marker, s.marker, s.place, s.object, s.unit, s.weather)
	fmt.Fprintf(&b, "User: Summarize the patch in two sentences using marker %s.\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Patch %s adds offset %d before modulus %d and keeps argv parsing in-process. Tests bind input %d to remainder %d without touching the %s ledger at %s.\n", s.marker, offset, modulus, input, want, s.object, s.place)
	return b.String()
}

func chatReviewCalibrationRecord(s synthScene, modulus int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Please review a pull request that adds dump and calibrate subcommands for probe %s. I am worried the author converted %s into a different unit at %s. Ask questions first if the diff is incomplete.\n", s.marker, s.unit, s.place)
	fmt.Fprintf(&b, "Assistant: Review notes for %s: I need to see whether argparse defaults freeze retries and timeout, whether dump prints the %s with unit %s, and whether calibrate writes marker %s into the filename. A conversion away from %s would be a blocking comment. I also want a test that parse_args receives a list, not a shell string. Housing %s should not become a --unit default.\n", s.marker, s.object, s.unit, s.marker, s.unit, s.material)
	fmt.Fprintf(&b, "User: Dump prints object=%s unit=%s place=%s. Calibrate reuses the same unit. There is no subprocess. What else should I test?\n", s.object, s.unit, s.place)
	fmt.Fprintf(&b, "Assistant: Add a test that a converted unit string is rejected, a test that marker %s appears in both subcommand outputs, and a test that modulus %d routing is not called from dump. Dump is a ledger copy; routing belongs in a separate helper. Conditions were recorded as %s and should stay a comment, not an argument. Do not treat %s as a test skip.\n", s.marker, modulus, s.quality, s.fault)
	fmt.Fprintf(&b, "User: If I land this today, what is the one-sentence merge note?\n")
	fmt.Fprintf(&b, "Assistant: Probe %s can dump and calibrate the %s at %s while leaving unit %s untouched and without spawning a shell.\n", s.marker, s.object, s.place, s.unit)
	fmt.Fprintf(&b, "User: Any leftover nit?\n")
	fmt.Fprintf(&b, "Assistant: Name the argparse prog probe-%s so logs grep. Do not abbreviate %s in help text; the ledger search depends on the full word. Weather %s is not a flag.\n", s.marker, s.place, s.weather)
	return b.String()
}

func chatPlanCalibrationRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Help me plan a three-step migration of ledger %s from a single file to a small SQLite table. We still sample the %s at %s in %s. I need a plan I can execute locally, not a cloud vendor pitch.\n", s.marker, s.object, s.place, s.unit)
	fmt.Fprintf(&b, "Assistant: Step 1: freeze the text ledger that contains marker %s and copy it. Step 2: create table field_ledger(marker, place, object_name, value_text, unit, recorded_at) and import without converting %s. Step 3: point the reader at SQLite but keep the text copy until two dumps match. Place %s and object %s are columns, not filenames hidden in a shell glob. Housing %s can be a nullable column later, not a unit alias.\n", s.marker, s.unit, s.place, s.object, s.material)
	fmt.Fprintf(&b, "User: What should I do if a row is missing unit %s during import?\n", s.unit)
	fmt.Fprintf(&b, "Assistant: Skip that row, record the line number, and keep marker %s in the error log. Do not default the unit. Do not %s a substitute from a neighboring row. Re-import after the source is fixed. Do not blame %s without a line number.\n", s.marker, s.action, s.fault)
	fmt.Fprintf(&b, "User: How do I prove the migration of %s succeeded?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: Count rows where marker=%s and unit=%s in both copies, then spot-check the %s at %s. If counts differ, the text file wins until you find the dropped line. Quality %s is not a column you should invent. Weather %s is not a WHERE clause.\n", s.marker, s.unit, s.object, s.place, s.quality, s.weather)
	fmt.Fprintf(&b, "User: Write the CREATE TABLE you would accept in review.\n")
	fmt.Fprintf(&b, "Assistant: CREATE TABLE field_ledger (\n  marker TEXT NOT NULL,\n  place TEXT NOT NULL,\n  object_name TEXT NOT NULL,\n  value_text TEXT NOT NULL,\n  unit TEXT NOT NULL,\n  recorded_at TEXT NOT NULL\n);\n-- marker %s, place %s, unit %s on every insert. Craft %s is not a column.\n", s.marker, s.place, s.unit, s.craft)
	return b.String()
}

func chatSQLFollowupCalibrationRecord(s synthScene) string {
	limit := 5 + s.index%8
	var b strings.Builder
	fmt.Fprintf(&b, "User: I ran the delta query for marker %s and got fewer than %d rows. The %s at %s should have a longer history in %s. What should I inspect before rewriting the SQL?\n", s.marker, limit, s.object, s.place, s.unit)
	fmt.Fprintf(&b, "Assistant: Inspect filters first: marker='%s', place='%s', unit='%s', object_name='%s'. A converted unit would drop rows. Then check timestamps so LAG has a predecessor. Only rewrite the ORDER BY if those filters are correct. Limit %d can wait. Housing %s is not in the WHERE unless you added it on purpose.\n", s.marker, s.place, s.unit, s.object, limit, s.material)
	fmt.Fprintf(&b, "User: The unit filter was wrong; someone stored 'secs' instead of %s. How do I repair without merging two instruments?\n", s.unit)
	fmt.Fprintf(&b, "Assistant: Do not UPDATE unit in place until you prove those rows are the same %s at %s. Export the 'secs' rows with marker %s, compare labels, and only then issue a narrow UPDATE ... WHERE marker='%s' AND unit='secs'. Leave other markers untouched. This is not %s; it is a label repair.\n", s.object, s.place, s.marker, s.marker, s.fault)
	fmt.Fprintf(&b, "User: Draft that UPDATE, and tell me what not to run.\n")
	fmt.Fprintf(&b, "Assistant: UPDATE field_ledger SET unit='%s' WHERE marker='%s' AND place='%s' AND unit='secs' AND object_name='%s'; Do not run UPDATE field_ledger SET unit='%s' without the marker predicate. Do not wrap the statement in a shell for loop.\n", s.unit, s.marker, s.place, s.object, s.unit)
	fmt.Fprintf(&b, "User: After the repair, what SELECT proves it?\n")
	fmt.Fprintf(&b, "Assistant: SELECT count(*) FROM field_ledger WHERE marker='%s' AND unit='%s'; and a second count for unit='secs' that should be zero for this marker. Keep the %s as object_name in both checks. Weather %s is not SQL.\n", s.marker, s.unit, s.object, s.weather)
	return b.String()
}

func chatOpsCalibrationRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: I started a local sampler for the %s at %s, log file probe-%s.log, unit %s. It is stuck on my machine. Walk me through a safe stop, then a restart checklist. This is my process.\n", s.object, s.place, s.marker, s.unit)
	fmt.Fprintf(&b, "Assistant: Identify the PID you own that opened probe-%s.log. Request a normal stop first and wait. If it stays up, force-stop that PID only. Confirm the log still ends with marker %s and unit %s. Then restart with the same argv list your supervisor already uses; do not invent a shell one-liner from the place name %s. Housing %s does not change the PID.\n", s.marker, s.marker, s.unit, s.place, s.material)
	fmt.Fprintf(&b, "User: After restart, the first line looks truncated. Do I keep it?\n")
	fmt.Fprintf(&b, "Assistant: Keep the truncated line as evidence, then start a new complete row. Do not %s a completed reading from the broken prefix. The %s under %s conditions still needs a full label plus unit %s. That truncation may be %s, not a new measurement.\n", s.action, s.object, s.quality, s.unit, s.fault)
	fmt.Fprintf(&b, "User: Give me a three-item restart checklist that mentions %s.\n", s.marker)
	fmt.Fprintf(&b, "Assistant: 1) Argv list includes probe %s. 2) Log path contains %s. 3) First complete post-restart row names place %s, object %s, and unit %s. No fourth item that converts units. Weather %s is gear, not argv.\n", s.marker, s.marker, s.place, s.object, s.unit, s.weather)
	fmt.Fprintf(&b, "User: If a teammate asks me to stop every sampler in %s, what do I say?\n", s.place)
	fmt.Fprintf(&b, "Assistant: Refuse the broadcast. Stop only the PID tied to marker %s on your workstation. Other samplers at %s are other people's processes unless they explicitly hand you ownership. Craft %s is not a warrant.\n", s.marker, s.place, s.craft)
	return b.String()
}

func chatMathCalibrationRecord(s synthScene) string {
	a := 6 + s.index%15
	b := 3 + s.index%9
	mod := a + 7
	var buf strings.Builder
	fmt.Fprintf(&buf, "User: Tutor me through (%d + %d) * %d versus %d + %d * %d. Then tell me why that is not a measurement of the %s at %s in %s. Marker %s.\n", a, b, b, a, b, b, s.object, s.place, s.unit, s.marker)
	fmt.Fprintf(&buf, "Assistant: (%d + %d) * %d = %d. %d + %d * %d = %d because multiplication binds first. Those are integers on a chalkboard, not %s. Marker %s keeps the tutoring ticket distinct from the ledger. Do not %s the product onto the instrument.\n", a, b, b, (a+b)*b, a, b, b, a+b*b, s.unit, s.marker, s.action)
	fmt.Fprintf(&buf, "User: Show the same identity with remainders modulo %d.\n", mod)
	fmt.Fprintf(&buf, "Assistant: (%d * %d) modulo %d is %d. Remainder is an index, not a converted %s. Place %s does not appear in the modulo. Fault to avoid: %s as if it were a remainder.\n", a, b, mod, (a*b)%mod, s.unit, s.place, s.fault)
	fmt.Fprintf(&buf, "User: One sentence that refuses to mix the lesson with the %s.\n", s.object)
	fmt.Fprintf(&buf, "Assistant: Ticket %s is arithmetic; the %s at %s stays in %s. Weather %s is not a factor.\n", s.marker, s.object, s.place, s.unit, s.weather)
	fmt.Fprintf(&buf, "User: Final check: is 2^%d equal to %d?\n", b, 1<<uint(b))
	fmt.Fprintf(&buf, "Assistant: 2^%d is %d. That identity is still not a reading of the %s. Marker %s.\n", b, 1<<uint(b), s.object, s.marker)
	return buf.String()
}

func chatRecipeCalibrationRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Write a kitchen procedure that is analogous to logging the %s at %s without converting %s. Marker %s is the batch name. This is cooking, not a lab cheat.\n", s.object, s.place, s.unit, s.marker)
	fmt.Fprintf(&b, "Assistant: Batch %s. Weigh the grain in grams if the scale says grams; do not silently switch to ounces. The analogy to the %s is: copy the unit on the scale plate. Place %s in this recipe is just the kitchen, not a field site. If the dough looks %s, that is a feel note, like weather %s, not a mass.\n", s.marker, s.object, s.place, s.quality, s.weather)
	fmt.Fprintf(&b, "User: Someone wrote the salt in a different unit. Repair?\n")
	fmt.Fprintf(&b, "Assistant: Do not convert in the margin. Weigh again in the original unit or mark the line incomplete. Marker %s stays on the card. That is the same rule as refusing to convert %s. Do not blame %s without a second weighing.\n", s.marker, s.unit, s.fault)
	fmt.Fprintf(&b, "User: What belongs on the serving note?\n")
	fmt.Fprintf(&b, "Assistant: Batch %s, kitchen %s, no converted masses. Housing %s is a bowl material, not a unit. Craft %s is baking, not a ledger rewrite.\n", s.marker, s.place, s.material, s.craft)
	fmt.Fprintf(&b, "User: One line that keeps the analogy honest.\n")
	fmt.Fprintf(&b, "Assistant: Cooking batch %s does not authorize changing the %s at %s from %s into another unit.\n", s.marker, s.object, s.place, s.unit)
	return b.String()
}

func chatEmailCalibrationRecord(s synthScene) string {
	var b strings.Builder
	fmt.Fprintf(&b, "User: Draft an internal email about delayed sampling of the %s at %s. Marker %s. Unit %s. No vendor pitch. Explain we will not convert to catch up.\n", s.object, s.place, s.marker, s.unit)
	fmt.Fprintf(&b, "Assistant: Subject: %s delayed, unit %s unchanged\n\nThe %s at %s was not %s on schedule because of %s weather and a bench note about %s. We will resume with the same unit %s. We will not convert stored rows to fill the hole. Marker %s remains the filename. Housing %s is unchanged.\n", s.marker, s.unit, s.object, s.place, s.action, s.weather, s.fault, s.unit, s.marker, s.material)
	fmt.Fprintf(&b, "User: Add a sentence for the night shift.\n")
	fmt.Fprintf(&b, "Assistant: If you only have the headline, open marker %s, copy unit %s, and do not average across the gap. Quality %s is not a severity code.\n", s.marker, s.unit, s.quality)
	fmt.Fprintf(&b, "User: What must not be in the cc line?\n")
	fmt.Fprintf(&b, "Assistant: A script that rewrites units. Craft %s people can be informed; they still cannot convert %s by email.\n", s.craft, s.unit)
	fmt.Fprintf(&b, "User: Closing.\n")
	fmt.Fprintf(&b, "Assistant: %s / %s / %s / %s. End of note.\n", s.marker, s.place, s.object, s.unit)
	return b.String()
}

func multilingualCalibrationRecord(s synthScene) string {
	scripts := multilingualCalibrationScripts()
	script := scripts[(s.index/len(syntheticCalibrationDomains))%len(scripts)]
	format := func(tmpl string) string {
		return fmt.Sprintf(tmpl, s.marker, s.place, s.object, s.unit, s.weather, s.material)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "User: %s\nAssistant: %s\nUser: %s\nAssistant: %s\nUser: %s\nAssistant: %s\n",
		format(script.u1), format(script.a1), format(script.u2), format(script.a2), format(script.u3), format(script.a3))
	return b.String()
}

type mlScript struct{ u1, a1, u2, a2, u3, a3 string }

func multilingualCalibrationScripts() []mlScript {
	// Placeholders: %[1]s marker, %[2]s place, %[3]s object, %[4]s unit, %[5]s weather, %[6]s material.
	return []mlScript{
		{
			u1: "Escribe una receta de lentejas para el lote %[1]s. La cocina está en %[2]s. No conviertas los gramos a onzas. El %[3]s del laboratorio no es un ingrediente; la unidad de la receta no es %[4]s.",
			a1: "Lote %[1]s: 250 g de lentejas, 1 litro de agua, 8 g de sal. Pesar en gramos si la báscula dice gramos. %[2]s es la cocina, no un campo. El tiempo %[5]s solo decide si se cierra la ventana. El bol es de %[6]s.",
			u2: "Alguien anotó la sal en onzas. ¿Puedo convertir para cerrar el lote %[1]s?",
			a2: "No. Vuelva a pesar o marque la línea incompleta. %[1]s no autoriza una conversión. %[3]s y %[4]s pertenecen a otro cuaderno.",
			u3: "Título de la tarjeta de cocina.",
			a3: "%[1]s, cocina %[2]s, gramos sin conversión, bol %[6]s.",
		},
		{
			u1: "Rédigez la fiche horaire %[1]s pour la halte de %[2]s. Les minutes restent des minutes. L'instrument %[3]s en %[4]s n'est pas un train. Météo %[5]s.",
			a1: "Fiche %[1]s: départ 07:12, arrivée 08:04, 52 minutes. %[2]s est une halte. On n'écrit pas ces minutes en %[4]s. Le carnet de %[3]s est un autre dossier. Le banc est en %[6]s seulement s'il s'agit d'un abri, pas d'une unité.",
			u2: "Puis-je convertir 52 minutes en secondes pour « simplifier » %[1]s ?",
			a2: "Non. L'horaire est en minutes. Convertir cacherait %[1]s. %[5]s peut retarder le train; cela n'autorise pas un changement d'unité.",
			u3: "Une ligne d'affichage.",
			a3: "%[1]s %[2]s 07:12-08:04, minutes, pas %[4]s.",
		},
		{
			u1: "Erstelle die Holzliste %[1]s für die Werkstatt in %[2]s. Längen in Centimetern lassen. %[3]s mit Einheit %[4]s gehört nicht auf diese Liste. Wetter %[5]s.",
			a1: "Liste %[1]s: 6 Latten zu 120 cm, Holzart %[6]s. %[2]s ist die Werkstatt. Keine Umrechnung in Zoll. Das Laborgerät %[3]s bleibt in seinem Heft mit %[4]s.",
			u2: "Darf ich 120 cm als 1,2 m schreiben, um %[1]s zu kürzen?",
			a2: "Nur wenn beide Schreibweisen auf derselben Zeile stehen. Allein 1,2 ohne cm löst die Bindung. %[5]s ist kein Maß.",
			u3: "Kopfzeile.",
			a3: "%[1]s, %[2]s, 120 cm, %[6]s, kein %[4]s.",
		},
		{
			u1: "Monte a tabela de preços %[1]s da feira em %[2]s. Preços em reais. O %[3]s em %[4]s é de outro caderno. Tempo %[5]s.",
			a1: "Tabela %[1]s: milho 4 reais/kg, feijão 8 reais/kg. %[2]s é o mercado. Não converter reais para outra moeda no rodapé. %[6]s é a banca, não uma unidade de laboratório.",
			u2: "Posso passar os preços para %[4]s para «alinhar» com o laboratório?",
			a2: "Não. %[4]s é do %[3]s. A feira %[1]s fica em reais. %[5]s pode esvaziar a feira; não muda a moeda.",
			u3: "Título da lousa.",
			a3: "%[1]s, %[2]s, reais/kg, banca %[6]s.",
		},
		{
			u1: "茶の計量ログ %[1]s を、測定値と単位を変えずに要約してください。場所は %[2]s。実験室の %[3]s（単位 %[4]s）は別ノートです。天気は %[5]s。",
			a1: "ログ %[1]s では、茶の測定値をグラムのまま残します。単位を換算しません。%[2]s は茶室です。%[3]s の単位 %[4]s を茶の行に混ぜません。器は %[6]s。欠測は欠測のままです。",
			u2: "第二の記録係がマーカー %[1]s を見ていないとき、場所 %[2]s だけから行を復元してよいですか。",
			a2: "いいえ。%[1]s が無い行は未完成です。%[2]s も %[3]s もマーカーの代わりにはなりません。単位 %[4]s を推測で補いません。天気 %[5]s も測定値ではありません。",
			u3: "記録 %[1]s の一行見出しを作ってください。",
			a3: "%[1]s：%[2]s の茶、測定値はグラム、単位を換算しない。実験室の %[3]s（%[4]s）は別行。",
		},
		{
			u1: "请为批次 %[1]s 写一份茶叶称重说明。地点 %[2]s。只保留克。实验室的 %[3]s（单位 %[4]s）不要写进这张卡。天气 %[5]s。",
			a1: "批次 %[1]s：茶叶按克记录。%[2]s 是茶室。不把克换成两。%[3]s 的 %[4]s 留在另一本笔记。托盘材料 %[6]s 只说明器具。",
			u2: "如果有人把克改成实验室的 %[4]s 来「对齐」，怎么办？",
			a2: "拒绝。%[1]s 与 %[3]s 不是同一行。天气 %[5]s 不是重量。",
			u3: "一句话标题。",
			a3: "%[1]s：%[2]s 茶叶，克，不换算，不是 %[4]s。",
		},
		{
			u1: "정류 안내 %[1]s 를 분 단위로 쓰세요. 장소 %[2]s. 실험실 %[3]s 의 단위 %[4]s 는 넣지 마세요. 날씨 %[5]s.",
			a1: "안내 %[1]s: %[2]s 에서 다음 정거장까지 4분. 분을 %[4]s 로 바꾸지 않습니다. %[3]s 는 다른 노트입니다. 벤치 재료 %[6]s 는 시설 설명입니다.",
			u2: "4분을 초로 바꿔 짧게 쓸까요?",
			a2: "같은 줄에 둘 다 쓰지 않으면 안 됩니다. %[1]s 는 분 안내입니다. %[5]s 는 지연 이유일 뿐 단위가 아닙니다.",
			u3: "전광판 한 줄.",
			a3: "%[1]s %[2]s 4분, 단위 분, %[4]s 아님.",
		},
		{
			u1: "صف مساحة الفناء في السجل %[1]s عند %[2]s بالأمتار. لا تحوّل إلى %[4]s. الجهاز %[3]s دفتر آخر. الطقس %[5]s.",
			a1: "السجل %[1]s: الفناء 12 م × 7 م. %[2]s هو الموقع. الأمتار تبقى أمتارًا. %[3]s بوحدة %[4]s لا يدخل في مساحة الفناء. المادة %[6]s للبلاط فقط.",
			u2: "هل أكتب 1200 سم بدل 12 م لاختصار %[1]s؟",
			a2: "فقط إن بقي المتر ظاهرًا في نفس السطر. الطقس %[5]s ليس طولًا.",
			u3: "عنوان اللوحة.",
			a3: "%[1]s، %[2]s، 12×7 م، ليست %[4]s.",
		},
		{
			u1: "वर्षा कार्ड %[1]s स्थान %[2]s के लिए मिलीमीटर में लिखें। प्रयोगशाला का %[3]s इकाई %[4]s इस कार्ड पर नहीं। मौसम %[5]s।",
			a1: "कार्ड %[1]s: 18 मिमी वर्षा। %[2]s गाँव है। मिमी को %[4]s में न बदलें। %[3]s अलग कॉपी है। बर्तन %[6]s केवल गेज स्टैंड है।",
			u2: "क्या 18 मिमी को 1.8 सेमी लिखकर %[1]s छोटा करूँ?",
			a2: "एक ही पंक्ति में दोनों हों तो चल सकता है। अकेली सेमी बाइंडिंग तोड़ती है। %[5]s माप नहीं है।",
			u3: "शीर्षक।",
			a3: "%[1]s, %[2]s, 18 मिमी, नहीं %[4]s।",
		},
		{
			u1: "Карточка часов %[1]s для партии в %[2]s. Минуты остаются минутами. Прибор %[3]s в %[4]s — другая тетрадь. Погода %[5]s.",
			a1: "Карточка %[1]s: контроль 15+10. %[2]s — клуб. Не переводите минуты в %[4]s. %[3]s не шахматные часы. Корпус %[6]s только про коробку часов.",
			u2: "Можно ли 15 минут записать как %[4]s, чтобы «согласовать лабораторию»?",
			a2: "Нет. %[1]s — шахматные минуты. %[5]s может отменить турнир, но не меняет единицу.",
			u3: "Строка на доске.",
			a3: "%[1]s %[2]s 15+10 мин, не %[4]s.",
		},
		{
			u1: "Zrób kartę drzewostanu %[1]s dla %[2]s. Średnice w centymetrach. %[3]s w %[4]s to inny notes. Pogoda %[5]s.",
			a1: "Karta %[1]s: 14 drzew, średnica 28 cm. %[2]s to oddział. Nie przeliczaj cm na %[4]s. %[3]s zostaje w laboratorium. Materiał %[6]s to taśma, nie jednostka.",
			u2: "Czy mogę zapisać 0,28 m zamiast 28 cm, żeby skrócić %[1]s?",
			a2: "Tylko z cm w tym samym wierszu. %[5]s to warunek pracy, nie średnica.",
			u3: "Nagłówek.",
			a3: "%[1]s, %[2]s, 28 cm, nie %[4]s.",
		},
	}
}

func longContextCalibrationRecord(s synthScene, places, objects, qualities, units []string) string {
	const entries = 28
	var b strings.Builder
	fmt.Fprintf(&b, "User: Dossier %s contains a sequence of field observations. Read all entries before answering; the question depends on the first and final entries. Do not answer from a middle page. Housing %s. Known fault somewhere in the file: %s.\n", s.marker, s.material, s.fault)
	for j := 0; j < entries; j++ {
		place := places[(s.index+j)%len(places)]
		object := objects[(s.index+3*j)%len(objects)]
		quality := qualities[(s.index+5*j)%len(qualities)]
		unit := units[(s.index+7*j)%len(units)]
		value := 10 + (s.index+11*j)%89
		switch j % 4 {
		case 1:
			fmt.Fprintf(&b, "Log %02d. %s — %s reads %d %s (%s). Marker %s. Copied once from the plate.\n", j+1, place, object, value, unit, quality, s.marker)
		case 2:
			fmt.Fprintf(&b, "Page %02d — at %s the %s still shows %d %s. Observer refused to average. Dossier %s. Note %s only as weather-of-the-line.\n", j+1, place, object, value, unit, s.marker, quality)
		case 3:
			fmt.Fprintf(&b, "Record %02d) %s / %s / %d %s. Dossier %s repeats so a torn copy still names it. Feel of the day: %s.\n", j+1, place, object, value, unit, s.marker, quality)
		default:
			fmt.Fprintf(&b, "Entry %02d-%s records the %s at %s as %d %s under %s conditions. The observer checked the label, copied the value once, and did not infer a missing measurement. Marker %s is repeated so a truncated paste still names the dossier.\n", j+1, s.marker, object, place, value, unit, quality, s.marker)
		}
	}
	firstObject := objects[s.index%len(objects)]
	lastObject := objects[(s.index+3*(entries-1))%len(objects)]
	fmt.Fprintf(&b, "Question: Which objects appear in the first and final entries, in that order?\nAssistant: The first entry uses the %s; the last entry uses the %s. Dossier %s requires both distant positions, not a guess from the middle entries. The unit on each row stays with that row; do not average them. Craft %s is the trade of the people who filed it, not a conversion license.\n", firstObject, lastObject, s.marker, s.craft)
	fmt.Fprintf(&b, "User: If entry 14 is missing from a copy of %s, may I reconstruct it from neighbors?\n", s.marker)
	fmt.Fprintf(&b, "Assistant: No. Dossier %s treats a hole as a hole. Neighbors have their own objects and units. Answer the first-and-last question from entries that are present; mark entry 14 as absent rather than interpolating. Do not blame %s without seeing the gap. Weather on a neighbor row is not a substitute reading.\n", s.marker, s.fault)
	return b.String()
}

func extendRecordIfShort(text string, s synthScene) string {
	n := 0
	for len(text) < minSyntheticRecordBytes && n < 8 {
		var b strings.Builder
		b.WriteString(text)
		if !strings.HasSuffix(text, "\n") {
			b.WriteByte('\n')
		}
		a := 11 + (s.index+n)%83
		delta := 3 + ((s.index+n)/5)%29
		switch n % 6 {
		case 1:
			fmt.Fprintf(&b, "User: Night-shift check %d on %s: the %s housing at %s still claims unit %s. Weather is %s. What is copied?\n", n+1, s.marker, s.material, s.place, s.unit, s.weather)
			fmt.Fprintf(&b, "Assistant: Copy unit %s and marker %s. Do not convert because the shift changed. Signed change if the display moved %d to %d is %d %s. Quality %s is not a reading. Defect rumor %s stays a rumor until logged.\n", s.unit, s.marker, a, a+delta, delta, s.unit, s.quality, s.fault)
		case 2:
			fmt.Fprintf(&b, "User: Editor note %d for %s: may I rewrite the %s row in another unit to match a spreadsheet?\n", n+1, s.marker, s.object)
			fmt.Fprintf(&b, "Assistant: No. Marker %s, place %s, unit %s. Spreadsheets do not license conversion. Craft %s is the trade, not the unit. Follow-up %d only restates the bind.\n", s.marker, s.place, s.unit, s.craft, n+1)
		case 3:
			fmt.Fprintf(&b, "User: Inventory aside %d: %d pieces of %s near the %s. Marker %s. Is that a measurement in %s?\n", n+1, 2+n, s.material, s.object, s.marker, s.unit)
			fmt.Fprintf(&b, "Assistant: No. Piece count %d is inventory. The %s stays in %s at %s. Marker %s. Do not %s the count onto the instrument.\n", 2+n, s.object, s.unit, s.place, s.marker, s.action)
		case 4:
			fmt.Fprintf(&b, "User: Routing aside %d for %s: %d mod %d. Why is the remainder not %s?\n", n+1, s.marker, a+delta, 17+n, s.unit)
			fmt.Fprintf(&b, "Assistant: Remainder %d is a bin. Physical unit remains %s. Marker %s at %s. Weather %s is not a modulus.\n", (a+delta)%(17+n), s.unit, s.marker, s.place, s.weather)
		case 5:
			fmt.Fprintf(&b, "User: Oral check %d: say the three bindings for %s aloud.\n", n+1, s.marker)
			fmt.Fprintf(&b, "Assistant: Marker %s. Place %s with object %s. Unit %s. Housing %s is furniture. Fault %s is a defect log, not a binding.\n", s.marker, s.place, s.object, s.unit, s.material, s.fault)
		default:
			fmt.Fprintf(&b, "User: Worked follow-up %d for marker %s: if the %s at %s moved from %d %s to %d %s after it was %s, what should the ledger say?\n", n+1, s.marker, s.object, s.place, a, s.unit, a+delta, s.unit, s.action)
			fmt.Fprintf(&b, "Assistant: The ledger for %s keeps the unit %s. Signed change is %d %s. Quality note: %s. Follow-up %d does not replace the earlier turns; it only restates the invariant so a truncated copy still binds %s to %s and refuses a unit conversion. Weather %s stays qualitative.\n", s.marker, s.unit, delta, s.unit, s.quality, n+1, s.marker, s.place, s.weather)
		}
		text = b.String()
		n++
	}
	return text
}
