# [←](Home.md) Описание полей моделирования

## Дерево `info-sim`
* `nfo_gen` — Farich Generator info
* `events` — number of events
* `momentum` — momentum of particles (MeV/c)
* `theta` — polar angle of particles (rad)
* `phi` — azimuthal angle of particles (rad)
* `zDis` —
* `info_rad` — Farich Radiator info
* `layers` — vector of layers
* `layers.first` — layers indeces of refraction
* `layers.second` — layers thickness (mm)
* `info_pmt` — Farich PMT info
* `name` — PMT name
* `num_side_x` — number of PMTs (x-axis)
* `num_side_y` — number of PMTs (y-axis)
* `gap` — size between PMTs, mm
* `size` — board size, mm
* `chip_num_size` — array size (num x num array)
* `chip_pitch` — distance between adjacent pixels
* `chip_size` — pixel size, mm
* `chip_offset` — chip offset
* `focal_length` — distance between Cherenkov radiator and PD&Electrocics, mm
* `trg_window` — trigger window, ns
* `origin_pos._0` — x coordinate of board center, mm
* `origin_pos._1` — y coordinate of board center, mm
* `origin_pos._2` — z coordinate of board center, mm

## Дерево `raw_data`
* `id_event` — event identifier
* `id_primary` — PDG code of initial particles
* `pos_primary._[0-2]` — coordinates of initial particles
* `dir_primary._[0-2]` — velocity directions (Vi/V) of initial particles
* `hits` — Farich Hit vector
* `hits.id_pmt` — id of pmt on board
* `hits.id_chip` — pixel id
* `hits.id_layer` — numbers of layers where photons were produced
* `hits.id_track` —
* `hits.id_track_parent` —
* `hits.id_hit` —
* `hits.id_fit` —
* `hits.wavelength` — wavelength of detected photon (nm)
* `hits.time` — time of detected photons (?s)
* `hits.pos_exact._[0-2]` — exact position of photons
* `hits.pos_chip._[0-2]` — pixel positions of photons
* `hits.pos_vertex._[0-2]` — vertex positions of photons
* `hits.dir_vertex._[0-2]` — vertex directions of photons
* `hits.theta.first` — theta of initial particle computed using hits, mean (rad) (?)
* `hits.theta.second` — theta of initial particle computed using hits, sigma (rad) (?)
* `hits.phi.first` — phi of initial particle computed using hits, mean (rad) (?)
* `hits.phi.second` — phi of initial particle computed using hits, sigma (rad) (?)


[**Вернуться на главную**](Home.md)