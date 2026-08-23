package quantize

import (
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
	"sync"
)

type calibrationPartition string

const (
	partitionCalibration calibrationPartition = "calibration"
	partitionSearch      calibrationPartition = "search"
	partitionValidation  calibrationPartition = "validation"
)

type calibrationRecord struct {
	ID        string
	Domain    string
	Source    string
	Partition calibrationPartition
	Text      string
}

type calibrationCorpus struct {
	Calibration []calibrationRecord
	Search      []calibrationRecord
	Validation  []calibrationRecord
}

func calibrationRecordPartition(id string) calibrationPartition {
	h := fnv.New32a()
	_, _ = h.Write([]byte(id))
	switch h.Sum32() % 10 {
	case 8:
		return partitionSearch
	case 9:
		return partitionValidation
	default:
		return partitionCalibration
	}
}

func appendCalibrationRecord(c *calibrationCorpus, r calibrationRecord) {
	switch r.Partition {
	case partitionSearch:
		c.Search = append(c.Search, r)
	case partitionValidation:
		c.Validation = append(c.Validation, r)
	default:
		c.Calibration = append(c.Calibration, r)
	}
}

// syntheticCalibrationDomains is the bundled Dynamic mix. Every partition
// (calibration, search, validation) must contain each name.
var syntheticCalibrationDomains = []string{
	"prose", "facts", "code", "multilingual", "structured-tool", "long-context", "refusal-adjacent", "chat",
}

const (
	syntheticCalibrationRecords = 4608
	minSyntheticRecordBytes     = 1800
)

type synthScene struct {
	place, object, action, quality, unit, marker string
	weather, material, craft, fault              string
	index                                        int
}

var (
	generatedCalibrationOnce   sync.Once
	generatedCalibrationCached calibrationCorpus
)

// generatedCalibrationCorpus produces original, deterministic examples rather
// than replaying a short seed text. Record IDs are hash-partitioned before any
// rendering, keeping search and validation examples out of calibration.
// Domain holdouts that still sit under minDomainHoldoutBytes steal extra
// unused calibration records of the same domain (never copied). Partitions
// are then round-robined by domain so mixed files do not clump.
func generatedCalibrationCorpus() calibrationCorpus {
	generatedCalibrationOnce.Do(func() {
		generatedCalibrationCached = buildGeneratedCalibrationCorpus()
	})
	return cloneCalibrationCorpus(generatedCalibrationCached)
}

func cloneCalibrationCorpus(c calibrationCorpus) calibrationCorpus {
	out := calibrationCorpus{
		Calibration: make([]calibrationRecord, len(c.Calibration)),
		Search:      make([]calibrationRecord, len(c.Search)),
		Validation:  make([]calibrationRecord, len(c.Validation)),
	}
	copy(out.Calibration, c.Calibration)
	copy(out.Search, c.Search)
	copy(out.Validation, c.Validation)
	return out
}

// Record builders live in calgen.go.

func topUpDomainHoldouts(c *calibrationCorpus) {
	seen := map[string]struct{}{}
	for _, recs := range [][]calibrationRecord{c.Calibration, c.Search, c.Validation} {
		for _, r := range recs {
			if r.Domain != "" {
				seen[r.Domain] = struct{}{}
			}
		}
	}
	domains := make([]string, 0, len(seen))
	for domain := range seen {
		domains = append(domains, domain)
	}
	sort.Strings(domains)
	for _, domain := range domains {
		topUpDomainPartition(c, domain, partitionSearch)
		topUpDomainPartition(c, domain, partitionValidation)
	}
}

func topUpDomainPartition(c *calibrationCorpus, domain string, dest calibrationPartition) {
	destRecords := &c.Validation
	if dest == partitionSearch {
		destRecords = &c.Search
	}
	for domainPartitionWrappedBytes(*destRecords, domain) < minDomainHoldoutBytes {
		if !moveOneCalibRecord(c, domain, destRecords, dest) {
			return
		}
	}
}

func moveOneCalibRecord(c *calibrationCorpus, domain string, dest *[]calibrationRecord, destPart calibrationPartition) bool {
	n := 0
	last := -1
	for i, r := range c.Calibration {
		if r.Domain == domain {
			n++
			last = i
		}
	}
	if n <= 1 || last < 0 {
		return false
	}
	rec := c.Calibration[last]
	rec.Partition = destPart
	c.Calibration = append(c.Calibration[:last], c.Calibration[last+1:]...)
	*dest = append(*dest, rec)
	return true
}

func domainPartitionWrappedBytes(records []calibrationRecord, domain string) int {
	var subset []calibrationRecord
	for _, r := range records {
		if r.Domain == domain {
			subset = append(subset, r)
		}
	}
	return len(wrapCalibration(chatFamilyPlain, renderCalibrationRecords(subset)))
}

func alphaMarker(n int) string {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"
	var buf [4]byte
	for i := len(buf) - 1; i >= 0; i-- {
		buf[i] = alphabet[n%len(alphabet)]
		n /= len(alphabet)
	}
	return string(buf[:])
}

func renderCalibrationRecords(records []calibrationRecord) string {
	records = interleaveRecordsByDomain(records)
	var b strings.Builder
	for i, r := range records {
		if i > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(strings.TrimSpace(r.Text))
	}
	if b.Len() > 0 {
		b.WriteByte('\n')
	}
	return b.String()
}

// interleaveRecordsByDomain round-robins records across first-seen domains
// so mixed imatrix text does not clump one domain for a long run. Same
// records, different order. A single-domain slice is returned unchanged.
func interleaveRecordsByDomain(records []calibrationRecord) []calibrationRecord {
	if len(records) < 2 {
		return records
	}
	buckets := map[string][]calibrationRecord{}
	var order []string
	for _, r := range records {
		d := r.Domain
		if d == "" {
			d = "_"
		}
		if _, ok := buckets[d]; !ok {
			order = append(order, d)
		}
		buckets[d] = append(buckets[d], r)
	}
	if len(order) < 2 {
		return records
	}
	out := make([]calibrationRecord, 0, len(records))
	for {
		added := false
		for _, d := range order {
			if len(buckets[d]) == 0 {
				continue
			}
			out = append(out, buckets[d][0])
			buckets[d] = buckets[d][1:]
			added = true
		}
		if !added {
			break
		}
	}
	return out
}

func interleaveCalibrationCorpus(c *calibrationCorpus) {
	c.Calibration = interleaveRecordsByDomain(c.Calibration)
	c.Search = interleaveRecordsByDomain(c.Search)
	c.Validation = interleaveRecordsByDomain(c.Validation)
}

// factCalibrationText is original Q&A used to exercise FFN key–value
// memories during llama-imatrix. Unsloth Dynamic 2.0's quality gap is
// largely a long mixed calibration set (300k–1.5M tokens). This is not
// that large, but it is unique tokens rather than looping 48KB of prose.
func factCalibrationText() string {
	var b strings.Builder
	b.Grow(96 << 10)
	b.WriteString("Bind these facts. Dates, names, and numbers stay glued to the event they belong to.\n\n")
	for _, e := range factEvents {
		fmt.Fprintf(&b, "User: %s\nAssistant: %s\n\n", e.q, e.a)
	}
	b.WriteString("Bind these facts in order. Do not swap years.\n")
	for i := 0; i+1 < len(factEvents); i += 2 {
		fmt.Fprintf(&b, "- %s\n- %s\n", factEvents[i].a, factEvents[i+1].a)
	}
	b.WriteString("\nUser: Which of each pair happened first?\nAssistant: ")
	for i := 0; i+1 < len(orderPairs); i += 2 {
		fmt.Fprintf(&b, "%s before %s. ", orderPairs[i], orderPairs[i+1])
	}
	b.WriteString("\n\n")
	b.WriteString(factCardsBlock)
	b.WriteString("\n")
	b.WriteString(factScienceBlock)
	b.WriteString("\n")
	b.WriteString(factGeoBlock)
	b.WriteString("\n")
	b.WriteString(factComputeBlock)
	b.WriteString("\n")
	return b.String()
}

type factQA struct{ q, a string }

// Public chronology in original sentences. Not copied encyclopedia text.
var factEvents = []factQA{
	{"When did Sputnik 1 launch, and what was it?", "Sputnik 1 launched on 4 October 1957. It was the first artificial satellite, a Soviet sphere that beeped on radio from low Earth orbit."},
	{"When did Yuri Gagarin orbit Earth?", "Yuri Gagarin completed one orbit on 12 April 1961 aboard Vostok 1. He was the first human in space. The flight lasted 108 minutes."},
	{"When did Alan Shepard fly, and was it orbital?", "Alan Shepard flew Freedom 7 on 5 May 1961. It was a 15-minute suborbital hop, not an orbit. John Glenn's Friendship 7 on 20 February 1962 was the first American orbital flight."},
	{"When did Valentina Tereshkova fly?", "Valentina Tereshkova flew Vostok 6 on 16 June 1963. She was the first woman in space. The mission lasted nearly three days."},
	{"When did Alexei Leonov walk in space?", "Alexei Leonov made the first spacewalk from Voskhod 2 on 18 March 1965. He was outside for about twelve minutes. His suit stiffened and he had trouble re-entering."},
	{"When did Gemini 8 dock, and what went wrong?", "Gemini 8 docked with an Agena on 16 March 1966, the first docking of two spacecraft. A stuck thruster sent them spinning. Neil Armstrong and David Scott aborted and splashed down early."},
	{"When did Apollo 1 kill its crew?", "Apollo 1 caught fire on the pad on 27 January 1967 during a plugs-out test. Virgil Grissom, Edward White, and Roger Chaffee died. The hatch opened inward and the cabin was pure oxygen."},
	{"When did Soyuz 1 crash?", "Soyuz 1 crashed on 24 April 1967. Vladimir Komarov died when the parachute failed. It was the first in-flight fatality of the space age."},
	{"When did Apollo 8 orbit the Moon?", "Apollo 8 entered lunar orbit on 24 December 1968. Frank Borman, James Lovell, and William Anders were the first humans to leave low Earth orbit and see the far side of the Moon."},
	{"When did Apollo 11 land, and who walked?", "Apollo 11 landed in the Sea of Tranquility on 20 July 1969. Neil Armstrong stepped onto the surface first; Buzz Aldrin followed. Michael Collins stayed in Columbia in lunar orbit."},
	{"When did Apollo 13 abort, and why?", "Apollo 13 aborted on the way to the Moon after an oxygen tank exploded on 13 April 1970. James Lovell, Jack Swigert, and Fred Haise used the lunar module as a lifeboat and returned on 17 April 1970."},
	{"When did Apollo 17 leave the Moon?", "Apollo 17, the last crewed Moon landing, lifted off from Taurus-Littrow on 14 December 1972. Eugene Cernan and Harrison Schmitt were the last to walk there. No crew has been back since."},
	{"When did Salyut 1 fly, and what happened to its crew?", "Salyut 1, the first space station, launched on 19 April 1971. The Soyuz 11 crew died on reentry on 30 June 1971 when a valve opened and the cabin depressurized."},
	{"When did Skylab launch?", "Skylab launched on 14 May 1973 on the last Saturn V. A micrometeoroid shield tore off at launch. The first crew repaired it and lived aboard for 28 days."},
	{"When did the Apollo-Soyuz docking happen?", "Apollo and Soyuz docked on 17 July 1975. Thomas Stafford and Alexei Leonov shook hands in orbit. It was the first international crewed docking."},
	{"When did Voyager 1 and 2 launch?", "Voyager 2 launched on 20 August 1977. Voyager 1 launched on 5 September 1977 and later overtook Voyager 2. Both used a rare outer-planet alignment."},
	{"When did the Space Shuttle first fly?", "Columbia flew STS-1 on 12 April 1981, twenty years to the day after Gagarin. John Young and Robert Crippen were the crew. It was the first reusable orbital spacecraft."},
	{"When was Challenger lost?", "Challenger was lost 73 seconds after launch on 28 January 1986. A solid-rocket O-ring failed in cold weather. All seven crew died, including teacher Christa McAuliffe."},
	{"When was Columbia lost?", "Columbia broke up on reentry on 1 February 2003. Wing leading-edge damage from foam at launch let hot gas in. All seven crew died."},
	{"When did Mir launch, and when did it reenter?", "Mir's core launched on 20 February 1986. It was occupied for most of the next 15 years. It was deorbited into the Pacific on 23 March 2001."},
	{"When did the ISS first launch a module?", "Zarya, the first ISS module, launched on 20 November 1998. Unity followed on STS-88 in December 1998. Permanent crews began with Expedition 1 on 2 November 2000."},
	{"When did Hubble launch, and when was its mirror fixed?", "Hubble launched on 24 April 1990 on STS-31. The primary mirror was figured wrong. STS-61 in December 1993 installed COSTAR and a new camera."},
	{"When did Pathfinder land on Mars?", "Mars Pathfinder landed on 4 July 1997. The Sojourner rover was the first wheeled vehicle on Mars. It used airbags, not a sky crane."},
	{"When did Spirit and Opportunity land?", "Spirit landed in Gusev crater on 4 January 2004. Opportunity landed in Meridiani Planum on 25 January 2004. Both outlived a 90-day plan; Opportunity last spoke in 2018."},
	{"When did Curiosity land?", "Curiosity landed in Gale crater on 6 August 2012 using a sky crane. It is a nuclear-powered rover, not solar. Perseverance landed in Jezero crater on 18 February 2021."},
	{"When did New Horizons fly past Pluto?", "New Horizons flew past Pluto on 14 July 2015. It launched on 19 January 2006. Charon, the large moon, was imaged the same day."},
	{"When did Cassini arrive at Saturn, and when did it end?", "Cassini entered Saturn orbit on 1 July 2004. Huygens landed on Titan on 14 January 2005. Cassini was destroyed in Saturn's atmosphere on 15 September 2017."},
	{"When did Juno arrive at Jupiter?", "Juno arrived at Jupiter on 5 July 2016. It is a polar orbiter with a radiation vault. It was not a lander."},
	{"When did China land on the lunar far side?", "Chang'e 4 landed in Von Kármán crater on 3 January 2019. It was the first landing on the far side. A relay satellite at Earth-Moon L2 carried the radio link."},
	{"When did Ingenuity first fly on Mars?", "Ingenuity first flew in Jezero crater on 19 April 2021. It was the first controlled extra-terrestrial aircraft. The air is about 1 percent of Earth's density."},
	{"When did JWST launch, and when were first images?", "JWST launched on 25 December 2021 on an Ariane 5. It orbits Sun-Earth L2. NASA released the first deep-field images on 12 July 2022."},
	{"When did the transistor appear relative to the IC and the microprocessor?", "The point-contact transistor is 1947 at Bell Labs. The integrated circuit is 1958-1959. A microprocessor as a CPU on one chip is 1971 (Intel 4004). Reverse that order and the causal story dies."},
	{"When was ENIAC finished versus the Manchester Baby?", "ENIAC was dedicated in 1946 and programmed with plugs and switches. The Manchester Baby ran a stored program on 21 June 1948. Stored-program and plugboard machines are not the same fact."},
	{"When did Sputnik 2 carry Laika?", "Sputnik 2 launched on 3 November 1957 with the dog Laika. There was no recovery plan. She was the first animal to orbit Earth."},
	{"When did Explorer 1 fly, and what did it find?", "Explorer 1 launched on 31 January 1958. It was the first US satellite. James Van Allen's Geiger counter found the radiation belts that now carry his name."},
	{"When did Luna 2 hit the Moon?", "Luna 2 impacted the Moon on 13 September 1959. It was the first human-made object to reach another celestial body. Luna 3 photographed the far side in October 1959."},
	{"When did Ranger 7 succeed?", "Ranger 7 returned close-up lunar photos on 31 July 1964 before impacting. Earlier Rangers had failed. Surveyor 1 soft-landed on 2 June 1966."},
	{"When did Venera 7 land on Venus?", "Venera 7 landed on Venus on 15 December 1970 and sent data from the surface. It was the first successful landing on another planet. The surface is about 460 C and 90 atmospheres."},
	{"When did Pioneer 10 fly past Jupiter?", "Pioneer 10 flew past Jupiter on 3 December 1973. Pioneer 11 did so on 3 December 1974 and later passed Saturn in 1979. They preceded the Voyagers."},
	{"When did Viking 1 land on Mars?", "Viking 1 landed in Chryse Planitia on 20 July 1976, seven years to the day after Apollo 11. Viking 2 landed in Utopia Planitia on 3 September 1976."},
	{"When did STS-107 fly relative to the Columbia accident?", "STS-107 launched on 16 January 2003 and was lost on 1 February 2003 during reentry. It was not a Hubble servicing mission. The foam strike happened at launch, not at landing."},
	{"When did Shenzhou 5 carry Yang Liwei?", "Shenzhou 5 launched on 15 October 2003. Yang Liwei was the first taikonaut. China became the third nation to launch its own crew."},
	{"When did SpaceX first land an orbital booster?", "Falcon 9 first landed an orbital first stage on 21 December 2015 at Cape Canaveral. A barge landing followed on 8 April 2016. Grasshopper hops were not orbital."},
	{"When did Crew Dragon first carry NASA astronauts?", "Demo-2 launched on 30 May 2020 with Doug Hurley and Bob Behnken. It was the first crewed orbital launch from the US since STS-135 in July 2011."},
	{"When was the first transatlantic telegraph cable a lasting success?", "A lasting Atlantic telegraph cable entered service in 1866 after an 1858 attempt failed. It is not the same year as the telephone (1876) or radio (the 1890s)."},
	{"When did Wright Flyer fly versus Lindbergh?", "The Wright Flyer flew on 17 December 1903 at Kitty Hawk. Lindbergh crossed the Atlantic solo on 20-21 May 1927. Alcock and Brown had already crossed nonstop in 1919."},
	{"When did penicillin become a usable drug versus Fleming's observation?", "Alexander Fleming noticed penicillin in 1928. Usable production and clinical use are a World War II story, peaking in 1944-1945. Observation and a pharmacy drug are different dates."},
	{"When was DNA's structure proposed?", "Watson and Crick proposed the double helix in 1953, using data from Franklin and Wilkins. Avery's 1944 work had already pointed to DNA as the genetic material. Those are not the same experiment."},
	{"When did the first human heart transplant happen?", "Christiaan Barnard transplanted a human heart on 3 December 1967 in Cape Town. Louis Washkansky lived 18 days. It was not the first organ transplant; kidneys came earlier."},
	{"When did Chernobyl versus Fukushima occur?", "Chernobyl's explosion was 26 April 1986. Fukushima Daiichi's accident followed the 11 March 2011 earthquake and tsunami. They are different reactor designs and different decades."},
	{"When did the Berlin Wall fall versus German reunification?", "The Wall's crossing opened on 9 November 1989. Official reunification is 3 October 1990. Opening a border and signing a state are not the same day."},
	{"When did the World Wide Web get proposed versus go public?", "Tim Berners-Lee proposed the Web at CERN in 1989 and the first site ran in 1990-1991. Mosaic popularized browsing in 1993. TCP/IP and the Web are not the same invention."},
	{"When did GPS become fully operational?", "GPS reached full operational capability in 1995 with 24 satellites. Selective availability was turned off on 1 May 2000. GLONASS is a separate Russian system."},
	{"When did the kilogram's definition change?", "The SI kilogram was redefined on 20 May 2019 using the Planck constant. The IPK metal cylinder in Paris is no longer the definition. The meter had already been tied to the speed of light in 1983."},
	{"How many seconds in an hour, and how many in a day?", "An hour is 3600 seconds. A day is 86400 seconds. A leap second is a UTC patch, not a change to the 86400 SI-second day used in physics."},
	{"What is the speed of light in vacuum, roughly?", "The speed of light in vacuum is exactly 299792458 m/s by definition of the meter. A useful round number is 3.00 times 10^8 m/s. It is not 3.00 times 10^6."},
	{"What is g near Earth's surface?", "Standard gravity is 9.80665 m/s^2. 9.8 m/s^2 is the usual engineering rounding. 10 m/s^2 is a back-of-envelope stand-in, not a measured local value."},
	{"What is absolute zero in C and F?", "Absolute zero is -273.15 C, which is -459.67 F, and 0 K. Water's triple point is 0.01 C, not 0 C. 0 C is the ice point at standard pressure."},
	{"What is the atomic number of carbon, nitrogen, and oxygen?", "Carbon is Z=6, nitrogen Z=7, oxygen Z=8. They sit in that order in the second period. Mixing Z with mass number is how people write C-12 as if it were Z."},
	{"What is Avogadro's number, roughly?", "Avogadro's number is about 6.022 times 10^23 entities per mole. A mole of carbon-12 atoms is 12 grams. 6.022 times 10^22 is off by ten."},
	{"How many chromosomes in a typical human somatic cell?", "46 chromosomes, 23 pairs. Gametes have 23 unpaired. Down syndrome is usually trisomy 21, which is 47 chromosomes, not 46."},
	{"What is the boiling point of water versus ethanol at 1 atm?", "Water boils at 100 C at 1 atm. Ethanol boils near 78 C. Confusing those two is a fluent error. Mercury boils far higher, near 357 C."},
	{"What is the chemical formula of water, methane, and ammonia?", "Water is H2O. Methane is CH4. Ammonia is NH3. CO2 is carbon dioxide, not carbon monoxide (CO). Mixing those formulae is a category error."},
	{"What year did the Titanic sink?", "Titanic sank on 15 April 1912 after hitting an iceberg on the night of 14 April. It was her maiden voyage. Lusitania was torpedoed in 1915; those are different ships and years."},
	{"When did World War I start and end?", "World War I ran from 1914 to 1918. The armistice was 11 November 1918. The US entered in 1917, not 1914. World War II is 1939-1945 in Europe."},
	{"When did World War II start in Europe versus the Pacific?", "Germany invaded Poland on 1 September 1939. Japan attacked Pearl Harbor on 7 December 1941. Those are not the same start date. VE Day is 8 May 1945; VJ Day is August 1945."},
	{"When did the first Moon landing happen relative to the Cuban Missile Crisis?", "The Cuban Missile Crisis was October 1962. Apollo 11 was July 1969. Seven years apart. Collapsing them into 'the sixties' loses the order."},
	{"When did the Chernobyl accident happen relative to the fall of the Soviet Union?", "Chernobyl was 1986. The Soviet Union dissolved in December 1991. Five years apart. One is a reactor; the other is a state."},
	{"When did UTF-8 get proposed versus when Unicode 1.0 shipped?", "Unicode 1.0 is 1991. UTF-8 was proposed in 1992 and later became the dominant web encoding. UCS-2 is not UTF-8; mixing them is how people claim a file is 'Unicode' without saying which encoding."},
	{"What is IPv4 localhost versus IPv6 localhost?", "IPv4 loopback is 127.0.0.1. IPv6 loopback is ::1. 0.0.0.0 is unspecified, not localhost. 255.255.255.255 is limited broadcast, not a host."},
	{"When did HTTP/1.1 get specified versus HTTP/2?", "HTTP/1.1 is RFC 2068 in 1997, later RFC 2616 in 1999. HTTP/2 is 2015. HTTP status 301 and 302 are redirects; 304 is not a redirect, it is not-modified."},
	{"What does HTTP 404 versus 410 versus 403 mean?", "404 is not found. 410 is gone, a deliberate removal. 403 is forbidden: the resource may exist but this caller may not read it. Collapsing all three into 'missing' loses the distinction."},
	{"When did Linux 1.0 ship versus Git's first public release?", "Linux 1.0 is March 1994. Git was announced in 2005. BitKeeper is not Git. Calling every kernel a 'Linux' in 1991 is the development year, not 1.0."},
	{"What is a kibibyte versus a kilobyte?", "A kibibyte (KiB) is 1024 bytes. A kilobyte (kB) in SI is 1000 bytes. Hard-drive marketing often uses 1000; RAM often uses 1024. Writing KB without saying which is how the same disk looks two sizes."},
}

var orderPairs = []string{
	"Sputnik 1 (1957)", "Gagarin (1961)",
	"Gagarin (1961)", "Glenn (1962)",
	"Leonov EVA (1965)", "Armstrong EVA on the Moon (1969)",
	"Apollo 8 lunar orbit (1968)", "Apollo 11 landing (1969)",
	"Apollo 11 (1969)", "Apollo 13 abort (1970)",
	"Apollo 17 (1972)", "STS-1 (1981)",
	"STS-1 (1981)", "Challenger (1986)",
	"Challenger (1986)", "Columbia (2003)",
	"Zarya (1998)", "Expedition 1 (2000)",
	"transistor (1947)", "integrated circuit (1958)",
	"integrated circuit (1958)", "microprocessor (1971)",
	"Wright Flyer (1903)", "Lindbergh (1927)",
	"Fleming (1928)", "wartime penicillin (1944)",
	"World War I (1914)", "World War II (1939)",
	"Unicode 1.0 (1991)", "UTF-8 proposal (1992)",
	"HTTP/1.1 (1997)", "HTTP/2 (2015)",
}

const factCardsBlock = `Chemistry cards. Z is the proton count, not the mass number.
User: Give Z for hydrogen, helium, lithium, iron, gold, uranium.
Assistant: H=1, He=2, Li=3, Fe=26, Au=79, U=92. Gold is not 26. Iron is not 79.

User: What is 2^10, 2^16, and 2^20?
Assistant: 2^10 is 1024. 2^16 is 65536. 2^20 is 1048576. 2^10 is not 1000. A KiB is 1024 bytes; a kB in SI is 1000.

User: How many minutes in a day, and how many hours in a week?
Assistant: A day is 1440 minutes. A week is 168 hours. 24*60 is 1440, not 2400. 7*24 is 168, not 178.

User: Convert 32 F, 0 C, 100 C, and 212 F.
Assistant: 32 F equals 0 C, freezing. 100 C equals 212 F, boiling at 1 atm. They are two physical points written two ways, not four temperatures.

User: What is the capital of France, Japan, Australia, Canada, and Brazil?
Assistant: Paris, Tokyo, Canberra, Ottawa, Brasília. Sydney is not Australia's capital. Rio de Janeiro is not Brazil's capital. Toronto is not Canada's.

User: Which river is longest, Nile or Amazon, and why is that a fight?
Assistant: Length rankings depend on the chosen source. The Nile is the classic school answer; the Amazon wins some modern measurements. Mixing "longest" with "most discharge" is wrong: the Amazon dominates discharge.

User: What is the tallest mountain above sea level versus from base to peak?
Assistant: Everest is the highest above sea level, 8848 or 8849 m depending on the survey. Mauna Kea is taller base-to-peak if you count the Hawaiian seafloor. Those are different questions.

ISO dates glued to events:
- 1957-10-04 Sputnik 1
- 1961-04-12 Vostok 1
- 1969-07-20 Apollo 11 landing
- 1970-04-13 Apollo 13 tank
- 1986-01-28 Challenger
- 1998-11-20 Zarya
- 2003-02-01 Columbia
- 2012-08-06 Curiosity
- 2021-12-25 JWST
`

const factScienceBlock = `Math cards. Exact integers, not vibes.
User: What is 13*17, 19*19, and 21*21?
Assistant: 13*17 is 221. 19*19 is 361. 21*21 is 441. 20*20 is 400; 19*19 is not 400.

User: Prime or not: 1, 2, 9, 11, 15, 49, 97.
Assistant: 1 is not prime. 2 is the even prime. 9=3*3, 15=3*5, 49=7*7 are composite. 11 and 97 are prime.

User: SI prefixes from milli to mega.
Assistant: milli 10^-3, centi 10^-2, kilo 10^3, mega 10^6. A milligram is not a megagram. Confusing milli and micro (10^-6) is a unit disaster.

User: What is RMSNorm versus LayerNorm in one sentence each?
Assistant: RMSNorm scales by rms and a gain, with no mean subtraction. LayerNorm subtracts the mean, divides by standard deviation, then applies gain and bias. RMSNorm stores fewer statistics.

Medicine-history cards.
User: Jenner, Pasteur, Fleming, Salk — what did each become famous for, and when roughly?
Assistant: Jenner published cowpox vaccination against smallpox in 1798. Pasteur's rabies vaccine is 1885. Fleming noticed penicillin in 1928. Salk's injected polio vaccine was widely used in 1955. Sabin's oral polio vaccine is a different product.

User: Insulin was isolated in which decade, and by whom in the usual textbook story?
Assistant: Banting and Best in 1921 in Toronto, with Collip and Macleod in the Nobel story. It is not a 1950s discovery. Synthetic recombinant insulin is decades later.

A grocery of constants:
- Planck's constant h is about 6.626 times 10^-34 J s
- Elementary charge e is about 1.602 times 10^-19 C
- Boltzmann k is about 1.381 times 10^-23 J/K
- Earth's mean radius is about 6371 km
- Moon mean distance is about 384000 km
- Astronomical unit is about 1.496 times 10^8 km
- Light-year is about 9.46 times 10^12 km
Do not swap the Moon's distance with an AU.
`

const factGeoBlock = `Geography cards. Capitals and rivers are not interchangeable with largest cities.
User: Capital of New Zealand, Kenya, Turkey, and Nigeria?
Assistant: Wellington, Nairobi, Ankara, Abuja. Auckland, Istanbul, and Lagos are larger cities, not the capitals.

User: Which ocean is largest, and which is deepest on average?
Assistant: The Pacific is largest by area. It also holds the Mariana Trench, the deepest known point, Challenger Deep, about 11 km. The Atlantic is not the largest.

User: Name three landlocked countries in Europe and three in Africa.
Assistant: Europe: Switzerland, Austria, Hungary (among others). Africa: Chad, Uganda, Ethiopia (Ethiopia has no coastline). Lesotho is landlocked inside South Africa. Mixing those continents is a map error.

User: When did the Panama Canal open versus Suez?
Assistant: Suez opened in 1869. Panama opened in 1914. Suez connects Mediterranean and Red Sea. Panama connects Atlantic and Pacific. Swapping them swaps oceans.

Code still belongs in calibration, or the matrix overfits chat.
func factYear(event string) int {
    switch event {
    case "sputnik1":
        return 1957
    case "apollo11":
        return 1969
    default:
        return 0
    }
}

SELECT name, year
FROM missions
WHERE year BETWEEN 1957 AND 1972
ORDER BY year, name;

Night trains still stop at the branch station even when nobody gets off.
The platform clock is three minutes fast. Arithmetic on a walk is how you
notice when a story and a landscape disagree: forty-eight posts at four
metres is one hundred ninety-two metres, not two hundred.
`

const factComputeBlock = `Computing cards. Names stay glued to encodings, status codes, and sizes.
User: Is UTF-8 the same as UTF-16, and what is a BOM for?
Assistant: UTF-8 and UTF-16 are different encodings of the same Unicode scalar values. A BOM is a signature some files use to mark byte order or UTF-8; it is not a character you should keep in a marker string. Latin-1 is not UTF-8.

User: What is the difference between a process ID and a port number?
Assistant: A PID names a process on one machine and is reused after exit. A TCP port names a bind on an interface. Copying a PID into a port field is a category error. Marker strings are neither.

User: How many bits in a byte, nibble, and uint32?
Assistant: A byte is 8 bits. A nibble is 4 bits. uint32 is 32 bits, which is 4 bytes, not 32 bytes. A 32-bit address space is not 32 bytes of RAM.

User: Git commit versus Git tag versus Git branch, in one sentence each.
Assistant: A commit is a snapshot with a parent. A tag is a name pointing at a commit, often immutable by convention. A branch is a moving name. Rewriting history does not change a measurement in another notebook.

User: JSON true, false, and null versus the strings "true", "false", and "null".
Assistant: Bare true/false/null are JSON literals. Quoted they are strings. Parsing "null" as a missing reading invents a value. Keep unit strings quoted; keep numbers unquoted only if they are numbers.

User: What is little-endian versus big-endian for a 16-bit word 0x1234?
Assistant: Little-endian stores 34 then 12. Big-endian stores 12 then 34. Mixing them looks like a swapped pair. That is a encoding fault, not a license to convert units.

User: Capital of South Korea, Argentina, Egypt, and Sweden?
Assistant: Seoul, Buenos Aires, Cairo, Stockholm. Mixing those with largest metro areas is less of a trap than mixing them with neighbors: Pyongyang is not Seoul.

User: Which writing system is Hangul, and which is Kanji?
Assistant: Hangul is the Korean alphabet. Kanji are Chinese characters used in Japanese. Hiragana and katakana are Japanese syllabaries. Calling all of them "Asian characters" unbinds the card.

A short table that must not reverse:
- 0x00 NUL, 0x0A LF, 0x0D CR
- HTTP 200 OK, 201 created, 204 no content
- IPv4 /24 is 256 addresses, /32 is one host
- A minute is 60 seconds; a leap second is UTC, not SI-day length
Do not store CR as LF or /24 as 24 hosts.
`
