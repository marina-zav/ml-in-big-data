-- 2.1
with max_table as (select max(scrobbles_lastfm) as max_scrobbles from hue__tmp_artists)
select distinct(artist_lastfm)
from hue__tmp_artists t
left join max_table m
where t.scrobbles_lastfm = m.max_scrobbles;

-- 2.2
select tag, count(*) as tag_count from hue__tmp_artists
lateral view explode(split(lower(tags_lastfm), '; ')) t1 as tag
where tag != ""
group by tag
order by tag_count desc
limit 1;

-- 2.3
with tags_table as
(
    select tag, count(*) as tag_count from hue__tmp_artists
    lateral view explode(split(lower(tags_lastfm), '; ')) t1 as tag
    where tag != ""
    group by tag
    order by tag_count desc
)

select artist_lastfm, max(listeners_lastfm) as max_l from 
    (
        select artist_lastfm,listeners_lastfm, tag from hue__tmp_artists
        lateral view explode(split(lower(tags_lastfm), '; ')) art as tag
    ) art_view
    
    left join tags_table
    on art_view.tag=tags_table.tag 
    group by artist_lastfm
    order by max_l desc
;


